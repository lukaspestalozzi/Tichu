import abc
from typing import Any, Dict, Tuple, Iterable
import logging
import numpy as np

import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Masking, Input, Merge
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.agents.sarsa import SarsaAgent
from rl.core import Processor, Env, MultiInputProcessor
from rl.policy import BoltzmannQPolicy, GreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from gym_tichu.envs.internals import (all_general_combinations_gen, BaseTichuState, PlayerAction, Card, PassAction,
                                      GeneralCombination, CardSet, CardRank)
from gym_tichu.envs import TichuSinglePlayerEnv

logger = logging.getLogger(__name__)

NBR_TICHU_ACTIONS = 258

# ### Processors ###
PASS_ACTION_NBR = 257
all_general_combinations = list(all_general_combinations_gen())
GENERALCOMBINATION_TO_NBR = {gcomb: idx for idx, gcomb in enumerate(all_general_combinations)}
NBR_TO_GENERALCOMBINATION = {v: k for k, v in GENERALCOMBINATION_TO_NBR.items()}


class Processor_56x5(Processor):

    def __init__(self):
        super().__init__()
        self.pass_action = PassAction(0)
        self.nbr_action_dict = {PASS_ACTION_NBR: self.pass_action}  # dictionary used to keep directly track of nbrs -> actual actions. This dict changes during the game

    @property
    def nb_inputs(self):
        return 2

    # My functions
    def encode_tichu_state(self, state: BaseTichuState)->Any:
        """
        Encodes the tichu-state for the NN,
        :param state: 
        :return: the encoded state and a dict mapping move nbrs to the action represented by that nbr.
        """
        # logger.debug("ecode tichu state: {}".format(state))

        def encode_cards(cards: Iterable[Card]):
            l = [False]*56
            if cards:
                for c in cards:
                    l[c.number] = True
            return l

        # encode handcards
        encoded = []
        for ppos in ((state.player_pos + k) % 4 for k in range(4)):
            # The players handcards are always in first position
            encoded.extend(encode_cards(state.handcards[ppos]))

        # encode trick on table
        encoded.extend(encode_cards(state.trick_on_table.last_combination))

        # encode possible actions
        encoded_gen_actions = [-500]*(len(all_general_combinations)+1)
        for action in state.possible_actions_list:
            if isinstance(action, PassAction):
                nbr = PASS_ACTION_NBR
            else:
                gcomb = GeneralCombination.from_combination(action.combination)
                try:
                    nbr = GENERALCOMBINATION_TO_NBR[gcomb]
                except KeyError:
                    logger.debug("comb: {}".format(action.combination))
                    logger.debug("gcomb: {}".format(gcomb))
                    logger.debug("dict: {}".format('\n'.join(map(str, GENERALCOMBINATION_TO_NBR.items()))))
                    raise
            encoded_gen_actions[nbr] = 0
            self.nbr_action_dict[nbr] = action

        assert len(encoded) == 56*5
        assert len(encoded_gen_actions) == 258
        enc = (np.array(encoded, dtype=bool), np.array(encoded_gen_actions))
        # logger.warning("enc: {}".format(enc))
        return enc

    def decode_action(self, action)->PlayerAction:
        try:
            t_action = self.nbr_action_dict[action]
        except KeyError:
            logger.debug("Process Action KeyError, There is probably no action possible.")  #action: {}, dict: {}".format(action, self.nbr_action_dict))
            t_action = action  # Happens when no action is possible
        # logger.debug('process_action: {} -> t_action {}'.format(action, t_action))
        return t_action

    def create_model(self, nbr_layers: int=2):
        """
        Creates the keras model for this processor with the given amount of hidden (dense) layers.
        :param nbr_layers: must be bigger or equal to 1
        :return: 
        """
        assert nbr_layers >= 1
        nbr_layers -= 1
        main_input_len = 56 * 5
        main_input = Input(shape=(main_input_len,), name='cards_input')
        main_line = Dense(NBR_TICHU_ACTIONS * 5, activation='elu')(main_input)
        for _ in range(nbr_layers):
            main_line = Dense(NBR_TICHU_ACTIONS, activation='elu')(main_line)

        # combine with the possible_actions input
        possible_actions_input = Input(shape=(NBR_TICHU_ACTIONS,), name='possible_actions_input')
        output = keras.layers.add([possible_actions_input, main_line])  # possible_actions_input is 0 where a legal move is, -500 where not. -> should set all illegal actions to -500

        model = Model(inputs=[main_input, possible_actions_input], outputs=[output])
        return model


    # processor functions
    def process_state_batch(self, state_batch):
        # logger.warning("state_batch: {}".format(str(state_batch)))
        input_batches = [[] for _ in range(self.nb_inputs)]
        for state in state_batch:
            processed_state = [[] for _ in range(self.nb_inputs)]
            for observation in state:
                # logger.warning("Observation: {}".format(str(observation)))
                assert len(observation) == self.nb_inputs
                for o, s in zip(observation, processed_state):
                    s.append(o)
            for idx, s in enumerate(processed_state):
                input_batches[idx].append(s)
            # logger.warning("input_batches: {}".format(input_batches))
            # logger.warning("input_batches len: {}".format(len(input_batches)))  # 7
        processed = [np.array(x) for x in input_batches]
        # logger.warning("processed: {}".format(str(processed)))
        ret = [np.squeeze(s, axis=(1,)) for s in processed]
        return ret


class Processor_56x5_2(Processor_56x5):

    def encode_tichu_state(self, state: BaseTichuState)->Any:
        """
        Encodes the tichu-state for the NN same as Processor_56x5, but there are 2 more bits present. 
        The first indicates whether the teammate finished first, the second whether the enemy finished first.
        """
        enc = super().encode_tichu_state(state)
        teammate_finished_first = len(state.ranking) and state.ranking[0] == (state.player_pos + 2) % 4
        enemy_finished_first = len(state.ranking) and state.ranking[0] != (state.player_pos + 2) % 4
        return (np.array(tuple(enc[0])+(teammate_finished_first, enemy_finished_first)),
                enc[1])

    def create_model(self, nbr_layers: int=2):
        """
        Creates the keras model for this processor with the given amount of hidden (dense) layers.
        :param nbr_layers: must be bigger or equal to 1
        :return: 
        """
        assert nbr_layers >= 1
        nbr_layers -= 1
        main_input_len = 56 * 5 + 2 # 2 bits more than Processor_56x5
        main_input = Input(shape=(main_input_len,), name='cards_input')
        main_line = Dense(NBR_TICHU_ACTIONS * 5 + 2, activation='elu')(main_input)
        for _ in range(nbr_layers):
            main_line = Dense(NBR_TICHU_ACTIONS, activation='elu')(main_line)

        # combine with the possible_actions input
        possible_actions_input = Input(shape=(NBR_TICHU_ACTIONS,), name='possible_actions_input')
        output = keras.layers.add([possible_actions_input, main_line])  # possible_actions_input is 0 where a legal move is, -500 where not. -> should set all illegal actions to -500

        model = Model(inputs=[main_input, possible_actions_input], outputs=[output])
        return model


class Processor_56x5_2_seperate(Processor_56x5_2):

    @property
    def nb_inputs(self):
        return 7

    def encode_tichu_state(self, state: BaseTichuState) -> Any:
        """
        Encodes the tichu-state for the NN same as Processor_56x5_2, but the 4 handcards and the trick on the table and the 2 bits are different inputs to the NN
        """
        enc = super().encode_tichu_state(state)
        sep = list()
        k = 0
        while k < len(enc[0]):
            sep.append(enc[0][k:k+56])
            k += 56
        assert len(sep) == 6
        assert len(sep[-1]) == 2
        assert all(len(p) == 56 for p in sep[:-1])

        ret = (*map(np.array, sep), enc[1])
        # logger.warning("shape of ret: {}".format(np.array(ret).shape))  #(7,)
        return ret

    def create_model(self, nbr_layers: int = 2):
        """
        Creates the keras model for this processor with the given amount of hidden (dense) layers.
        :param nbr_layers: must be bigger or equal to 1
        :return: 
        """
        assert nbr_layers >= 1
        nbr_layers -= 1

        ipt0 = Input(shape=(56,), name='input0')
        line0 = Dense(56, activation='elu')(ipt0)

        ipt1 = Input(shape=(56,), name='input1')
        line1 = Dense(56, activation='elu')(ipt1)

        ipt2 = Input(shape=(56,), name='input2')
        line2 = Dense(56, activation='elu')(ipt2)

        ipt3 = Input(shape=(56,), name='input3')
        line3 = Dense(56, activation='elu')(ipt3)

        ipttot = Input(shape=(56,), name='inputTot')
        line4 = Dense(56, activation='elu')(ipttot)

        iptbits = Input(shape=(2,), name='inputBits')
        line5 = Dense(2, activation='elu')(iptbits)

        # concatenate the lines
        main_line = keras.layers.concatenate([line0, line1, line2, line3, line4, line5])

        for _ in range(nbr_layers):
            main_line = Dense(NBR_TICHU_ACTIONS, activation='elu')(main_line)

        # combine with the possible_actions input
        possible_actions_input = Input(shape=(NBR_TICHU_ACTIONS,), name='possible_actions_input')
        output = keras.layers.add([possible_actions_input,
                                   main_line])  # possible_actions_input is 0 where a legal move is, -500 where not. -> should set all illegal actions to -500

        model = Model(inputs=[ipt0, ipt1, ipt2, ipt3, ipttot, iptbits, possible_actions_input], outputs=[output])
        return model


class Processor_17x5_2(Processor_56x5):

    @property
    def nb_inputs(self):
        return 2

    def encode_tichu_state(self, state: BaseTichuState) -> Any:
        def encode_cards(cards: CardSet):
            rank_dict = cards.rank_dict()
            return [len(rank_dict.get(rank, [])) for rank in CardRank]

        # encode handcards
        encoded = []
        for ppos in ((state.player_pos + k) % 4 for k in range(4)):
            # The players handcards are always in first position
            encoded.extend(encode_cards(state.handcards[ppos]))

        # encode trick on table
        lc = state.trick_on_table.last_combination
        if lc:
            encoded.extend(encode_cards(lc.cards))
        else:
            encoded.extend([0]*17)

        # add 2 bits
        teammate_finished_first = len(state.ranking) and state.ranking[0] == (state.player_pos + 2) % 4
        enemy_finished_first = len(state.ranking) and state.ranking[0] != (state.player_pos + 2) % 4
        encoded.append(teammate_finished_first)
        encoded.append(enemy_finished_first)

        # encode possible actions
        encoded_gen_actions = [-500]*(len(all_general_combinations)+1)
        for action in state.possible_actions_list:
            if isinstance(action, PassAction):
                nbr = PASS_ACTION_NBR
            else:
                gcomb = GeneralCombination.from_combination(action.combination)
                try:
                    nbr = GENERALCOMBINATION_TO_NBR[gcomb]
                except KeyError:
                    logger.debug("comb: {}".format(action.combination))
                    logger.debug("gcomb: {}".format(gcomb))
                    logger.debug("dict: {}".format('\n'.join(map(str, GENERALCOMBINATION_TO_NBR.items()))))
                    raise
            encoded_gen_actions[nbr] = 0
            self.nbr_action_dict[nbr] = action

        assert len(encoded) == 17*5 + 2
        assert len(encoded_gen_actions) == 258
        enc = (np.array(encoded), np.array(encoded_gen_actions))
        # logger.warning("enc: {}".format(enc))
        return enc

    def create_model(self, nbr_layers: int = 2):
        assert nbr_layers >= 1
        nbr_layers -= 1
        main_input_len = 17 * 5 + 2
        main_input = Input(shape=(main_input_len,), name='cards_input')
        main_line = Dense(NBR_TICHU_ACTIONS * 5, activation='elu')(main_input)
        for _ in range(nbr_layers):
            main_line = Dense(NBR_TICHU_ACTIONS, activation='elu')(main_line)

        # combine with the possible_actions input
        possible_actions_input = Input(shape=(NBR_TICHU_ACTIONS,), name='possible_actions_input')
        output = keras.layers.add([possible_actions_input, main_line])  # possible_actions_input is 0 where a legal move is, -500 where not. -> should set all illegal actions to -500

        model = Model(inputs=[main_input, possible_actions_input], outputs=[output])
        return model


class Processor_17x5_2_seperate(Processor_17x5_2):
    @property
    def nb_inputs(self):
        return 7

    def encode_tichu_state(self, state: BaseTichuState) -> Any:
        enc = super().encode_tichu_state(state)
        sep = list()
        k = 0
        while k < len(enc[0]):
            sep.append(enc[0][k:k + 17])
            k += 17
        assert len(sep) == 6, len(sep)
        assert len(sep[-1]) == 2
        assert all(len(p) == 17 for p in sep[:-1])

        ret = (*map(np.array, sep), enc[1])
        # logger.warning("shape of ret: {}".format(np.array(ret).shape))  #(7,)
        return ret

    def create_model(self, nbr_layers: int = 2):
        """
        Creates the keras model for this processor with the given amount of hidden (dense) layers.
        :param nbr_layers: must be bigger or equal to 1
        :return: 
        """
        assert nbr_layers >= 1
        nbr_layers -= 1
        nbr_cards = 17

        ipt0 = Input(shape=(nbr_cards,), name='input0')
        line0 = Dense(nbr_cards, activation='elu')(ipt0)

        ipt1 = Input(shape=(nbr_cards,), name='input1')
        line1 = Dense(nbr_cards, activation='elu')(ipt1)

        ipt2 = Input(shape=(nbr_cards,), name='input2')
        line2 = Dense(nbr_cards, activation='elu')(ipt2)

        ipt3 = Input(shape=(nbr_cards,), name='input3')
        line3 = Dense(nbr_cards, activation='elu')(ipt3)

        ipttot = Input(shape=(nbr_cards,), name='inputTot')
        line4 = Dense(nbr_cards, activation='elu')(ipttot)

        iptbits = Input(shape=(2,), name='inputBits')
        line5 = Dense(2, activation='elu')(iptbits)

        # concatenate the lines
        main_line = keras.layers.concatenate([line0, line1, line2, line3, line4, line5])

        for _ in range(nbr_layers):
            main_line = Dense(NBR_TICHU_ACTIONS, activation='elu')(main_line)

        # combine with the possible_actions input
        possible_actions_input = Input(shape=(NBR_TICHU_ACTIONS,), name='possible_actions_input')
        output = keras.layers.add([possible_actions_input,
                                   main_line])  # possible_actions_input is 0 where a legal move is, -500 where not. -> should set all illegal actions to -500

        model = Model(inputs=[ipt0, ipt1, ipt2, ipt3, ipttot, iptbits, possible_actions_input], outputs=[output])
        return model


def make_dqn_rl_agent(processor: Processor_56x5, nbr_layers=2, enable_dueling_network: bool=False, enable_double_dqn: bool=True):
    """
    
    :param processor: 
    :param nbr_layers: 
    :param enable_dueling_network:
    :param enable_double_dqn:
    :return: 
    """

    model = processor.create_model(nbr_layers=nbr_layers)
    test_policy = GreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)

    dqn_agent = DQNAgent(model=model, nb_actions=NBR_TICHU_ACTIONS, memory=memory, nb_steps_warmup=100,
                         target_model_update=1e-2, test_policy=test_policy, processor=processor,
                         enable_dueling_network=enable_dueling_network, enable_double_dqn=enable_double_dqn)
    dqn_agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn_agent


def make_sarsa_rl_agent(processor: Processor_56x5, nbr_layers=2):
    model = processor.create_model(nbr_layers=nbr_layers)
    test_policy = GreedyQPolicy()

    sarsa_agent = SarsaAgent(model=model, nb_actions=NBR_TICHU_ACTIONS, nb_steps_warmup=10, gamma=0.99, test_policy=test_policy, processor=processor)
    sarsa_agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return sarsa_agent


class TichuSinglePlayerTrainEnv(Env, metaclass=abc.ABCMeta):

    def __init__(self, processor: Processor_56x5, verbose: bool=True):
        super().__init__()
        self.game = TichuSinglePlayerEnv()
        self.processor = processor

    def reset(self):
        state = self.game._reset()
        return self.processor.encode_tichu_state(state)

    def step(self, action):
        playeraction = self.processor.decode_action(action)
        state, r, done, info = self.game._step(playeraction)
        return self.processor.encode_tichu_state(state), r, done, info

    def render(self, mode='human', close=False):
        pass

    def configure(self, *args, **kwargs):
        return self.game.configure(*args, **kwargs)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

