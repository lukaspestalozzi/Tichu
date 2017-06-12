
import gym
import logging
from typing import Union, Tuple, Any, List
from profilehooks import timecall

from .internals import *
from .internals.error import IllegalActionError, LogicError


logger = logging.getLogger(__name__)


class TichuMultiplayerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, illegal_move_mode: str='raise', verbose: bool=True):
        """
        :param illegal_move_mode: 'raise' or 'loose'. If 'raise' an exception is raised, 'loose' and the team looses 200:0
        :param verbose: if True, logs to the info log, if false, logs to the debug log
        """
        assert illegal_move_mode in ['raise'], "'loose' is not yet implemented"  # ['raise', 'loose']

        super().__init__()

        self._current_state = None
        self.verbose = verbose

    def _step(self, action: Any)-> Tuple[TichuState, Tuple[int, int, int, int], bool, dict]:
        # logger.debug("_step with action {}".format(action))

        state = self._current_state.next_state(action)
        self._current_state = state

        done = state.is_terminal()
        points = state.count_points() if done else (0, 0, 0, 0)
        if done:
            state = state.change(history=state.history.add_last_state(state))
        return state, points, done, dict()  # state, reward, done, info

    def _reset(self)->InitialState:
        self._current_state = InitialState()
        return self._current_state

    def _render(self, mode='human', close=False):
        # print("RENDER: ", self._current_state)
        pass

    def _log(self, message, *args, **kwargs):
        if self.verbose:
            logger.info(message, *args, **kwargs)
        else:
            logger.debug(message, *args, **kwargs)


class TichuSinglePlayerEnv(TichuMultiplayerEnv):
    """
    Environment for one player. The other players can be set with the 'configure' method
    """

    def __init__(self, verbose: bool=True):
        """
        :param verbose: if True, logs to the info log, if false, logs to the debug log
        """

        self._agents = (None, None, None, None)  # set with the 'configure' method
        super().__init__()

    @timecall(immediate=False)
    def _step(self, action: PlayerAction)-> Tuple[BaseTichuState, int, bool, dict]:
        assert self._agents[2] is not None

        try:
            state, reward, done, info = super()._step(action)
            # logger.debug("Legal Action! {}".format(action))
        except IllegalActionError:
            logger.debug("Illegal Action! {}, legal are: {}".format(action, self._current_state.possible_actions_list))
            return self._current_state, -500, True, {'illegalAction': action}

        player_state, reward, done, info = self._forward_to_player(state)

        assert done or player_state.player_pos == 0
        # if done:
        #     logger.debug("TichuSinglePlayerAgainstRandomEnv, Final State: {}".format(state))

        assert done == player_state.is_terminal()
        assert done or player_state.player_pos == 0, str(player_state)
        return player_state, reward, done, dict()  # state, reward, done, info

    def _reset(self)->BaseTichuState:
        # init
        _ = super()._reset()
        # NO grand tichu
        _ = super()._step({})
        # No (normal) tichu now
        _ = super()._step({})
        # No trading cards
        state, _, _, _ = super()._step([])
        # forward to player
        player_state, reward, done, info = self._forward_to_player(state)
        return player_state

    def _forward_to_player(self, state: BaseTichuState)->Tuple[BaseTichuState, int, bool, dict]:
        """
        :return: The next state in which the player 0 can play a Combiantion.
        """
        # logger.debug("Forwarding to player 0")

        if state.is_terminal():
            # logger.debug("State is already terminal, Nothing to forward.")
            return state, state.count_points()[0], True, dict()

        curr_state = state
        curr_reward = (0, 0, 0, 0)
        done = False
        info = dict()

        first_action = state.possible_actions_list[0]
        # Note: for both tichu and wish action, state.player_pos is not the same as action.player_pos, it is the pos of the next player to play a combination

        while not isinstance(first_action, (PassAction, PlayCombination)) or first_action.player_pos != 0:
            # logger.debug("state: {}".format(state))
            # No TICHU
            if isinstance(first_action, TichuAction):
                no_tichu_action = next(filter(lambda act: act.announce is False, curr_state.possible_actions_list))
                curr_state, curr_reward, done, info = super()._step(no_tichu_action)

            # No WISH
            elif isinstance(first_action, WishAction):
                no_wish_action = WishAction(player_pos=first_action.player_pos, wish=None)
                curr_state, curr_reward, done, info = super()._step(no_wish_action)

            # TRICK ENDS
            elif isinstance(first_action, WinTrickAction):
                curr_state, curr_reward, done, info = super()._step(first_action)

            # Play Combination
            elif isinstance(first_action, (PassAction, PlayCombination)):
                assert curr_state.player_pos != 0
                # other agents choose action

                action = self._agents[curr_state.player_pos].action(state=curr_state)
                curr_state, curr_reward, done, info = super()._step(action)

            else:
                raise LogicError()

            if done:
                # logger.debug("State is terminal -> break out of forward to player")
                # logger.debug("Final State: {}".format(curr_state))
                break

            first_action = curr_state.possible_actions_list[0]

        assert done or curr_state.player_pos == 0
        assert done == curr_state.is_terminal()
        return curr_state, curr_reward[0], done, info

    def configure(self, *args, other_agents: Tuple[Any, Any, Any], **kwargs):
        """
        :param other_agents: the 3 agents to play against. Note that other_agents[1] is the teammate.
        :return: 
        """
        assert len(other_agents) == 3
        self._agents = (None,) + tuple(other_agents)
