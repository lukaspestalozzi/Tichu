
import datetime
from math import ceil

import rl
from profilehooks import timecall
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

import logginginit
from typing import Union, Optional, List, Set, Tuple, Collection, Any, Iterable, Dict
from collections import OrderedDict

from gym_tichu.envs.internals import (wishable_card_ranks, CardTrade)
from gym_tichu.envs.internals.utils import check_param, make_sure_path_exists
from rl.core import Processor, Env
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy

from .keras_rl_utils import (make_dqn_rl_agent, make_sarsa_rl_agent, Processor_56x5, Processor_56x5_2_seperate,
                             Processor_17x5_2, Processor_17x5_2_seperate)


from .minimax import *
from .mcts import *
from . import strategies

logger = logging.getLogger(__name__)
human_logger = logginginit.CONSOLE_LOGGER

# TODO rename to DefaultAgent

class DefaultGymAgent(object):
    """
    Returns the first action presented to it
    """

    def __init__(self, announce_tichu: strategies.TichuStrategyType=strategies.never_announce_tichu_strategy,
                       announce_grand_tichu: strategies.TichuStrategyType=strategies.never_announce_tichu_strategy,
                       make_wish: strategies.WishStrategyType=strategies.random_wish_strategy,
                       trade: strategies.TradingStrategyType=strategies.random_trading_strategy,
                       give_dragon_away: strategies.DragonAwayStrategyType=strategies.give_dragon_to_the_right_strategy):

        self.announce_tichu = announce_tichu
        self.announce_grand_tichu = announce_grand_tichu
        self.make_wish = make_wish
        self.trade = trade
        self.give_dragon_away = give_dragon_away

    @property
    def info(self):
        return "{me.__class__.__name__}, Takes always the first action".format(me=self)

    def action(self, state):
        logger.debug("BaseAgent chooses from actions: {}".format([str(a) for a in state.possible_actions()]))
        return next(state.possible_actions_list)


# ################## RANDOM ####################
class RandomAgent(DefaultGymAgent):
    """
    Returns one of the possible actions at random
    """

    @property
    def info(self):
        return "{me.__class__.__name__}, Makes a random action".format(me=self)

    def action(self, state):
        logger.debug("RandomAgent chooses from actions: {}".format([str(a) for a in state.possible_actions_list]))
        return random.choice(state.possible_actions_list)


class BalancedRandomAgent(RandomAgent):
    """
    Chooses one combination type first, then returns one of the possible actions of this type (at random)
    """

    @property
    def info(self):
        return "{me.__class__.__name__}, Chooses first a Combination category and then an action".format(me=self)

    def action(self, state):
        # logger.debug("BalancedRandomAgent chooses from actions: {}".format([str(a) for a in state.possible_actions_list]))
        d = defaultdict(list)
        for action in state.possible_actions_list:
            try:
                d[action.combination.__class__].append(action)
            except AttributeError:
                d[action.__class__].append(action)

        return random.choice(d[random.choice(list(d.keys()))])


# ################## Minimax ####################
class MinimaxAgent(DefaultGymAgent):

    def __init__(self, depth: int=4):
        super().__init__()
        self._search = Minimax()
        self.depth = depth

    @property
    def info(self):
        return "{me.__class__.__name__}, depth: {me.depth}".format(me=self)

    def action(self, state: TichuState)->PlayerAction:
        if len(state.possible_actions_set) == 1:
            act = next(iter(state.possible_actions_set))
            logger.debug("There is only one possible action: {}".format(act))
            return act
        else:
            logger.debug("Starting Minimax search with depth {}".format(self.depth))
            res = self._search.search(root_state=state, max_depth=self.depth)
            return res

    def __str__(self):
        return "{me.__class__.__name__}({me._search.__class__.__name__}, depth: {me.depth})".format(me=self)


# ################## MCTS ####################
class BaseMonteCarloAgent(DefaultGymAgent):

    def __init__(self, search_algorithm: Ismcts, iterations: int=100, max_time: float=float('inf'), cheat: Union[bool, float]=False):
        super().__init__()
        assert cheat in [True, False] or 0 <= cheat <= 1
        self._search = search_algorithm
        self.iterations = iterations
        self.cheat = cheat
        self.max_time = max_time

    @property
    def info(self):
        return "{me.__class__.__name__}, iterations: {me.iterations}, cheat: {me.cheat}, search: {me._search.info}".format(me=self)

    def action(self, state: TichuState)->PlayerAction:
        if len(state.possible_actions_set) == 1:
            act = next(iter(state.possible_actions_set))
            logger.debug("There is only one possible action: {}".format(act))
            return act
        else:
            return self._search.search(root_state=state,
                                       observer_id=state.player_pos,
                                       iterations=self.iterations,
                                       max_time=self.max_time,
                                       cheat=self.cheat)

    def __str__(self):
        return "{me.__class__.__name__}({me._search.__class__.__name__}, {me.iterations}, {me.cheat})".format(me=self)


# ################## HUMAN ####################
class HumanInputAgent(DefaultGymAgent):

    def __init__(self, position: int):
        super().__init__(announce_tichu=HumanInputAgent.announce_tichu,
                         announce_grand_tichu=HumanInputAgent._announce_grand_tichu,
                         make_wish=HumanInputAgent.make_wish,
                         trade=HumanInputAgent.trade,
                         give_dragon_away=HumanInputAgent.give_dragon_away)

        assert position in range(4)
        self._position = position
        human_logger.info(f"You are Player Number {self._position}")
        human_logger.info(f"Your Teammate is {(self._position + 2) % 4}")
        self._can_announce_tichu = True

    @property
    def info(self):
        return "{me.__class__.__name__}, Asks a human to choose actions".format(me=self)

    @staticmethod
    def give_dragon_away(state: TichuState, player: int)->int:
        e_left = (player  - 1) % 4
        e_right = (player + 1) % 4
        human_logger.info("You won the Trick with the Dragon. Therefore you have to give it away.")
        human_logger.info("Whom do you want to give it?")
        human_logger.info("The enemy to the left ({e_left}) has {el_cards} cards, the enemy to the right ({e_right}) has {er_cards} cards left".format(e_left=e_left, el_cards=len(state.handcards[e_left]),
                                                                                                                                           e_right=e_right, er_cards=len(state.handcards[e_right])))
        pl_pos = int(HumanInputAgent.ask_user_to_choose_one_of([str((player + 1) % 4), str((player - 1) % 4)]))
        return pl_pos

    @staticmethod
    def make_wish(state: TichuState, player: int)->CardRank:
        human_logger.info(f"You played the MAHJONG ({str(Card.MAHJONG)}) -> You may wish a card Value:")
        answ = HumanInputAgent.ask_user_to_choose_one_of([cv.name for cv in wishable_card_ranks])
        cv = CardRank.from_name(answ)
        return cv

    @staticmethod
    def trade(state: TichuState, player: int)->Tuple[Card, Card, Card]:
        human_logger.info("Random Cards are being Traded for you!")
        return strategies.random_trading_strategy(state, player)

    @staticmethod
    def _announce_grand_tichu(state: TichuState, already_announced: Set[int], player: int):
        return HumanInputAgent._ask_announce_tichu(state.handcards[player], announced_tichu=set(), announced_grand_tichu=already_announced, grand=True)

    @staticmethod
    def announce_tichu(state: TichuState, already_announced: Set[int], player: int):
        return HumanInputAgent._ask_announce_tichu(state.handcards[player], announced_tichu=already_announced, announced_grand_tichu=state.announced_grand_tichu)

    def action(self, state: TichuState)->PlayerAction:
        if state.trick_on_table.is_empty():
            human_logger.info("Your turn to Play first. There is no Trick on the Table.")

        else:
            human_logger.info("Your turn to Play:")
            human_logger.info("Current Trick on the Table: "+str(state.trick_on_table))
            human_logger.info("On the Table is following Combination: {}".format(str(state.trick_on_table.last_combination)))

        human_logger.info(f"You have following cards: {list(map(str, (sorted(state.handcards[self._position]))))}")
        if state.wish:
            human_logger.info(f"If you can you have to play a {state.wish} because this is the wish of the MAHJONG.")

        human_logger.info("The other player have this many cards:")
        for k in range(4):
            human_logger.info(f"Player {k}: {len(state.handcards[k])}{' (Your teamate)' if (k+2) % 4 == self._position else ' (You)' if k==self._position else ''}")

        possible_actions = state.possible_actions_list
        pass_ = False
        possible_combinations = []
        comb_action_dict = {}
        for action in possible_actions:
            if isinstance(action, PassAction):
                pass_ = "PASS"
                comb_action_dict[pass_] = action
            else:
                comb_action_dict[action.combination] = action
                possible_combinations.append(action.combination)

        possible_combinations = sorted(possible_combinations, key=lambda comb: (len(comb), comb.height))
        if pass_:
            possible_combinations.insert(0, pass_)

        comb = self.ask_user_to_choose_with_numbers(possible_combinations)

        return comb_action_dict[comb]

    def traded_cards_received(self, card_trades: CardTrade):
        print("You received following Cards during swapping:")
        for sw in card_trades:
            relative_pos = 'Teammate'
            if sw.from_ == (self._position + 1) % 4:
                relative_pos = 'Enemy Right'
            elif sw.from_ == (self._position - 1) % 4:
                relative_pos = 'Enemy Left'
            print(f"From {sw.from_}({relative_pos}) you got {sw.card}")
        self.ask_user_for_keypress()

    @staticmethod
    def _ask_announce_tichu(handcards: HandCards, announced_tichu: Set[int], announced_grand_tichu: Set[int], grand: bool=False):
        if len(announced_grand_tichu):
            human_logger.info("Following Players announced a Grand Tichu: {}".format(announced_grand_tichu))
        if len(announced_tichu):
            human_logger.info("Following Players announced a Normal Tichu: {}".format(announced_tichu))

        human_logger.info(f"Your handcards are: {handcards}")
        answ = HumanInputAgent.ask_user_yes_no_question("Do you want to announce a {}Tichu?".format("Grand-" if grand else ""))
        assert answ in {True, False}
        return answ

    @staticmethod
    def ask_user_to_choose_one_of(possible_answers: List[Any])->Any:
        options_dict = OrderedDict([(k, k) for k in possible_answers])
        answer, _ = HumanInputAgent.ask_user_to_choose_from_options(options_dict)
        return answer

    @staticmethod
    def ask_user_to_choose_with_numbers(options: Iterable[Any])->Any:
        """
        Asks the user to choose one of the options, but makes it easier by numbering the options and the user only has to enter the number.
        :param options:
        :return: The chosen option
        """
        options_dict = OrderedDict(sorted((nr, o) for nr, o in enumerate(options)))
        inpt, comb = HumanInputAgent.ask_user_to_choose_from_options(options_dict)
        return comb

    @staticmethod
    def ask_user_yes_no_question(question: str)->bool:
        """
        Asks the user to answer the question.
        :param question: string
        :return: True if the answer is 'Yes', False if the answer is 'No'.
        """
        inpt, val = HumanInputAgent.ask_user_to_choose_from_options({'y': 'Yes', 'n': 'No'}, text=question + " \n{}\n")
        return val == 'Yes'

    @staticmethod
    def ask_user_to_choose_from_options(answer_option_dict,
                                        text: str='Your options: \n{}\n',
                                        no_option_text: str="You have no options, press Enter to continue.\n",
                                        one_option_text: str="You have only one option ({}), press Enter to choose it.\n"):

        """
        1 Displays the mapping from keys to values in option_answer_dict. (If key and value are the same, only key is displayed)

        2 and asks the user to choose one of the key values (waiting for user input is blocking).

        3 Repeats the previous steps until the input matches with one of the keys.

        4 Then returns the chosen key and corresponding value.

        :param answer_option_dict: Dictionary of answer -> option mappings. where answer is the text the user has to input to choose the option.
        :param text: Text displayed when there are 2 or more options. It must contain one {} where the possible options should be displayed.
        :param no_option_text: Text displayed when there is no option to choose from.
        :param one_option_text: Text displayed when there is exactly 1 option to choose from. It must contain one {} where the possible option is displayed.
        :return: The key, value pair (as tuple) chosen by the user. If the dict is empty, returns (None, None).
        """

        if len(answer_option_dict) == 0:
            HumanInputAgent.ask_user_for_input(no_option_text)
            return None, None
        elif len(answer_option_dict) == 1:
            HumanInputAgent.ask_user_for_input(one_option_text.format(next(iter(answer_option_dict.values()))))
            return next(iter(answer_option_dict.items()))
        else:
            answer_option_dict = OrderedDict({str(k): o for k, o in answer_option_dict.items()})
            opt_string = "\n".join("'"+str(answ)+"'"+(" for "+str(o) if answ != o else "") for answ, o in answer_option_dict.items())
            possible_answers = {str(k) for k in answer_option_dict}
            while True:
                inpt = HumanInputAgent.ask_user_for_input(text.format(opt_string))
                if inpt in possible_answers:
                    human_logger.info("You chose: {} -> {}".format(inpt, answer_option_dict[inpt]))
                    return inpt, answer_option_dict[inpt]
                else:
                    human_logger.info("Wrong input, please try again.")

    @staticmethod
    def ask_user_for_keypress():
        """
        Asks the user to press any key.
        :return: The input given by the user
        """
        return HumanInputAgent.ask_user_for_input("Press Enter to continue.")

    @staticmethod
    def ask_user_for_input(text: str)->str:
        """
        Displays the text and waits for user input
        :param text:
        :return: The user input
        """
        human_logger.info(text)
        return input()


# ################## LEARNING / DQN ####################
class RLAgent(BalancedRandomAgent):

    def __init__(self,  agent: rl.core.Agent, weights_file: Optional[str]):
        """
        
        :param agent: 
        :param weights_file: Either a file with the weights, or None
        """
        super().__init__()
        self.agent = agent
        self._weights_file = weights_file
        if weights_file:
            print("{} loading the weights from {}".format(self.__class__.__name__, weights_file))
            try:
                self.agent.load_weights(weights_file)
            except OSError as oserr:
                logger.exception(oserr)
                logger.error("Could not load file. Continuing with previous weights.")
                self._weights_file = None

    @property
    def processor(self):
        return self.agent.processor

    @property
    def info(self):
        if self._weights_file:
            return "{me.__class__.__name__}, with weights {me._weights_file}".format(me=self)
        else:
            return "{me.__class__.__name__} without weights file".format(me=self)

    def action(self, state: TichuState) -> PlayerAction:
        if len(state.possible_actions_list) == 1:
            logger.debug("LearningAgent has only 1 possible action: {}".format(state.possible_actions_list[0]))
            return state.possible_actions_list[0]
        elif not isinstance(state.possible_actions_list[0], (PlayCombination, PassAction)):
            a = super().action(state=state)
            logger.debug("LearningAgent got a non PlayCombination or PassAction, choosing randomly. {}".format(a))
            return a
        else:
            processed_state = self.agent.processor.encode_tichu_state(state)
            a = self.agent.forward(processed_state)
            action = self.agent.processor.decode_action(a)
            logger.debug("Q-agent chooses action: {}".format(action))
            return action

    def train(self, env: Env, base_folder: str, nbr_steps: int=1000):
        """
        Trains the agent
        :param env: The environment to train on
        :param base_folder: 
        :param nbr_steps: 
        """
        timef = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # filenames
        weights_out_filename = 'dqn_weights_{t}_trained_{nbr}.h5f'.format(t=timef, nbr=nbr_steps)
        checkpoint_weights_filename = 'dqn_checkpoint_weights_'+timef+'_{step}.h5f'
        log_filename = 'trainlog_dqn_{}.json'.format(timef)

        # full path of files
        template = '{folder}/{filename}'
        weights_out_file        = template.format(folder=base_folder, filename=weights_out_filename)
        checkpoint_weights_file = template.format(folder=base_folder, filename=checkpoint_weights_filename)
        log_file                = template.format(folder=base_folder, filename=log_filename)

        # Make sure the files/folders exists
        make_sure_path_exists(base_folder)
        for fn in [weights_out_file, log_file]:
            if not os.path.exists(fn):
                with open(fn, "w"):
                    pass

        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_file, interval=ceil(nbr_steps//5))]  # 5 checkpoints
        callbacks += [FileLogger(log_file, interval=ceil(nbr_steps//100))]  # update 100 times

        # set LinearAnnealedPolicy policy such that tau reaches min value at 75% of nbr_steps
        self.agent.policy = LinearAnnealedPolicy(BoltzmannQPolicy(clip=(-500, 300)), attr='tau', value_max=0.8, value_min=0.3, value_test=0.01, nb_steps=ceil(nbr_steps*0.8))
        # self.agent.policy = BoltzmannQPolicy(clip=(-500, 300), tau=0.1)

        self.agent.fit(env, nb_steps=nbr_steps, visualize=False, verbose=0, nb_max_start_steps=0, callbacks=callbacks)

        logger.info("saving the weights to {}".format(weights_out_file))
        self.agent.save_weights(weights_out_file, overwrite=True)


# make different RL agents

class DQNAgent2L_56x5(RLAgent):

    def __init__(self,  weights_file: Optional[str]=None):
        rlagent = make_dqn_rl_agent(processor=Processor_56x5(), nbr_layers=2)
        wfile = weights_file if weights_file else '{}/agent_weights/dqn_56x5_2layers.h5f'.format(os.path.dirname(os.path.realpath(__file__)))
        super().__init__(agent=rlagent, weights_file=wfile)


class DQNAgent4L_56x5(RLAgent):

    def __init__(self,  weights_file: Optional[str]=None):
        rlagent = make_dqn_rl_agent(processor=Processor_56x5(), nbr_layers=4)
        wfile = weights_file if weights_file else '{}/agent_weights/dqn_56x5_4layers.h5f'.format(os.path.dirname(os.path.realpath(__file__)))
        super().__init__(agent=rlagent, weights_file=wfile)


class DQNAgent2L_56x5_2_sep(RLAgent):

    def __init__(self,  weights_file: Optional[str]=None):
        rlagent = make_dqn_rl_agent(processor=Processor_56x5_2_seperate(), nbr_layers=2)
        wfile = weights_file if weights_file else '{}/agent_weights/dqn_56x5_2_seperate.h5f'.format(os.path.dirname(os.path.realpath(__file__)))
        super().__init__(agent=rlagent, weights_file=wfile)


class DQNAgent2L_17x5_2(RLAgent):

    def __init__(self,  weights_file: Optional[str]=None):
        rlagent = make_dqn_rl_agent(processor=Processor_17x5_2(), nbr_layers=2)
        wfile = weights_file if weights_file else '{}/agent_weights/dqn_17x5_2layers.h5f'.format(os.path.dirname(os.path.realpath(__file__)))
        super().__init__(agent=rlagent, weights_file=wfile)


class DQNAgent2L_17x5_2_sep(RLAgent):

    def __init__(self,  weights_file: Optional[str]=None):
        rlagent = make_dqn_rl_agent(processor=Processor_17x5_2_seperate(), nbr_layers=2)
        wfile = weights_file if weights_file else '{}/agent_weights/dqn_17x5_2_seperate_2layers.h5f'.format(os.path.dirname(os.path.realpath(__file__)))
        super().__init__(agent=rlagent, weights_file=wfile)


class SarsaAgent2L_56x5(RLAgent):

    def __init__(self,  weights_file: Optional[str]=None):
        rlagent = make_sarsa_rl_agent(processor=Processor_56x5(), nbr_layers=2)
        wfile = weights_file if weights_file else '{}/agent_weights/sarsa_56x5_2layers.h5f'.format(os.path.dirname(os.path.realpath(__file__)))
        super().__init__(agent=rlagent, weights_file=wfile)


# ################## Composite ####################


class DoubleAgent(DefaultGymAgent):
    """
    Agent that is composed of 2 agents.
    
    The first agent chooses the action if the player has more handcards than the switch_length, the second otherwise.
    """

    def __init__(self, agent1: DefaultGymAgent, agent2: DefaultGymAgent, switch_length: int=5):
        check_param(switch_length in range(1, 15))
        super().__init__()
        self.first_agent = agent1
        self.second_agent = agent2
        self.switch_len = switch_length

    @property
    def info(self):
        return "{me.__class__.__name__}, agent1: {me.first_agent.info}, switch_length: {me.switch_len}, agent2: {me.second_agent.info}".format(me=self)

    def action(self, state: TichuState)->PlayerAction:
        my_handcards = state.handcards[state.player_pos]
        if len(my_handcards) > self.switch_len:
            return self.first_agent.action(state)
        else:
            return self.second_agent.action(state)
