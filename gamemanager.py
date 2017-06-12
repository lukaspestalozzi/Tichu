from collections import namedtuple
from time import time
import logging
from typing import Tuple, List, Any
import datetime

import gym

import gym_tichu  # needed to register the environment
import itertools
from gym_tichu.envs.internals import *
from gym_tichu.envs.internals.utils import error_logged, time_since
from profilehooks import timecall

import logginginit
from gym_agents.agents import HumanInputAgent

logger = logging.getLogger(__name__)
console_logger = logginginit.CONSOLE_LOGGER

GameOutcome = namedtuple("GameOutcome", ["points", "history"])


class TichuGame(object):
    """
    Plays a 4-player game
    """

    def __init__(self, agent0, agent1, agent2, agent3):
        env = gym.make('tichu_multiplayer-v0')
        # env = wrappers.Monitor(env, "/home/lu/semester_project/tmp/gym-results")
        self.env = env

        self._agents = (agent0, agent1, agent2, agent3)

    @property
    def agents(self):
        return self._agents

    def start_game(self, target_points=1000)->Tuple[Tuple[int, int], List[Any]]:
        """
        Starts the tichu game
        Returns a tuple containing the points the two teams made
        """
        with error_logged(logger):  # log all raised errors
            start_t = time()
            console_logger.info("Starting game... target: {}".format(target_points))

            round_histories = list()
            nbr_errors = 0
            nbr_errors_to_ignore = 99

            points = (0, 0)

            while points[0] < target_points and points[1] < target_points:
                # run rounds until there is a winner
                try:
                    round_points, round_history = self._start_round()
                    round_histories.append(round_history)
                    points = (round_points[0] + points[0], round_points[1] + points[1])
                    console_logger.warning("=========================================")
                    console_logger.warning("Intermediate Result: {}".format(points))
                    console_logger.warning("=========================================")
                except Exception as err:
                    # log the 10 first errors, but continue with next round.
                    nbr_errors += 1
                    if nbr_errors > nbr_errors_to_ignore:
                        raise
                    else:
                        logger.error("There was en error while running a round. Next {} errors will be ignored.".format(nbr_errors_to_ignore-nbr_errors))
                        logger.exception(err)

            console_logger.info("[GAME END] Game ended: {p} [Nbr_Errors: {nbr_errs}, Time: {time_passed}]".format(p=points, nbr_errs=nbr_errors, time_passed=time_since(since=start_t)))

        return GameOutcome(points, round_histories)

    def _start_round(self)->Tuple[Tuple[int, int], Any]:
        start_t = time()
        console_logger.info("[ROUND START] Start round...")

        curr_state, reward, done, info = self._setup_round()
        assert reward == (0, 0, 0, 0)
        assert done is False
        logger.debug("Set up state: {}".format(curr_state))

        console_logger.info("Player {} has the MAHJONG and can start.".format(curr_state.player_pos))

        while not done:
            loop_start_t = time()

            first_action = curr_state.possible_actions_list[0]
            chosen_action = None
            # Note: for both tichu and wish action, state.player_pos is not the same as action.player_pos, it is the pos of the next player to play a combination
            # TODO assert that all actions are of the same type
            # TODO find the appropriate action and call next_state with it?

            # TICHU
            if isinstance(first_action, TichuAction):
                announce = self._agents[first_action.player_pos].announce_tichu(state=curr_state, already_announced=curr_state.announced_tichu, player=first_action.player_pos)
                if announce:
                    console_logger.info("[TICHU] announced by {}".format(first_action.player_pos))
                else:
                    console_logger.debug("{} does not announce a tichu".format(first_action.player_pos))
                chosen_action = TichuAction(player_pos=first_action.player_pos, announce_tichu=announce)

            # WISH
            elif isinstance(first_action, WishAction):
                wish = self._agents[first_action.player_pos].make_wish(state=curr_state, player=first_action.player_pos)
                console_logger.info("[WISH] {} by {}".format(wish, first_action.player_pos))
                chosen_action = WishAction(player_pos=first_action.player_pos, wish=wish)

            # TRICK ENDS
            elif isinstance(first_action, WinTrickAction):
                console_logger.info("[WIN TRICK] by {}".format(first_action.player_pos))
                chosen_action = first_action
                if isinstance(first_action, GiveDragonAwayAction):
                    to_player = self._agents[first_action.player_pos].give_dragon_away(state=curr_state, player=first_action.player_pos)
                    console_logger.info("[DRAGON AWAY] {} gives the dragon trick to {}".format(first_action.player_pos, to_player))
                    chosen_action = GiveDragonAwayAction(player_from=first_action.player_pos, player_to=to_player, trick=first_action.trick)

            # PLAY COMBINATION OR PASS
            else:
                current_player = curr_state.player_pos
                console_logger.info("[NEXT TO PLAY] Player{}'s turn to play on: {}".format(current_player, curr_state.trick_on_table.last_combination))
                console_logger.debug("with handcards: {}".format(curr_state.handcards[current_player]))

                # the agent chooses an action
                chosen_action = self._agents[current_player].action(curr_state)

                if isinstance(chosen_action, PassAction):
                    console_logger.info("[PASS] {}".format(current_player))
                else:
                    console_logger.info("[PLAY] {} plays {}".format(current_player, chosen_action))

                console_logger.debug("[Time: {}]".format(time_since(since=loop_start_t)))

            # APPLY THE ACTION
            curr_state, reward, done, info = self.env.step(chosen_action)
            if len(curr_state.handcards[current_player]) == 0:
                console_logger.info("[FINISH] player {} just finished. -> new ranking: {}".format(current_player, curr_state.ranking))

            console_logger.debug("Trick on table is now: {}".format(curr_state.trick_on_table))
            console_logger.debug("Current Handcards: \n{}".format(curr_state.handcards))

            console_logger.info("Combination on table is {}: {}".format('now' if isinstance(chosen_action, PlayCombination) or curr_state.trick_on_table.is_empty() else 'still', curr_state.trick_on_table.last_combination))

        console_logger.debug("Final State: {}".format(curr_state))
        points = (reward[0], reward[1])
        console_logger.warning("[ROUND END] Round ended: ranking: {}, outcome: {} [Time: {}]".format(curr_state.ranking, points, time_since(since=start_t)))
        return GameOutcome(points, curr_state.history)

    def _setup_round(self)->Tuple[TichuState, Tuple[int, int, int, int], bool, dict]:
        """
        Sets up the round until the first player can play a combinaiton
        :return: a 4-tuple(state, reward_vector, done, info_dict)
        """
        s_8cards = self.env.reset()
        # grand tichu
        announced_gt = set()
        for ppos, agent in enumerate(self._agents):
            if agent.announce_grand_tichu(state=s_8cards, already_announced=set(announced_gt), player=ppos):
                announced_gt.add(ppos)
        console_logger.info("[GRAND TICHUS]: {}".format(announced_gt))
        console_logger.debug("handcards: {}".format(s_8cards.handcards))

        s_14cards, _, _, _ = self.env.step(announced_gt)

        # players may announce (normal) tichu now
        announced_t = set()
        for ppos, agent in enumerate(self._agents):
            if agent.announce_tichu(state=s_14cards, already_announced=set(announced_t), player=ppos):
                announced_t.add(ppos)
        console_logger.info("[TICHUS]: {}".format(announced_t))
        console_logger.debug("handcards: {}".format(s_14cards.handcards))

        s_before_trading, _, _, _ = self.env.step(announced_t)

        # trade cards
        traded_cards = list()
        for ppos, agent in enumerate(self._agents):
            trade_cards = agent.trade(state=s_before_trading, player=ppos)
            # some checks
            assert len(set(trade_cards)) == len(trade_cards)  # cant trade the same card twice
            # TODO assert s_before_trading.has_cards(player=ppos, cards=trade_cards)  # player must have the card

            traded_cards.append(CardTrade(from_=ppos, to=(ppos - 1) % 4, card=trade_cards[0]))
            traded_cards.append(CardTrade(from_=ppos, to=(ppos + 2) % 4, card=trade_cards[1]))
            traded_cards.append(CardTrade(from_=ppos, to=(ppos + 1) % 4, card=trade_cards[2]))

        # Notify the human players about the cards they received  # TODO this is a hack
        for pos, agent in enumerate(self._agents):
            if isinstance(agent, HumanInputAgent):
                agent.traded_cards_received(filter(lambda tc: tc.to == pos, traded_cards))

        return self.env.step(traded_cards)
