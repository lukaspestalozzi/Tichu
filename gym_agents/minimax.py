import logging
from operator import itemgetter
from typing import Optional, Union, Hashable, NewType, TypeVar, Tuple, List, Dict, Iterable, Generator, Set, FrozenSet
from time import time
from profilehooks import timecall, profile

from gym_tichu.envs.internals import (TichuState, PlayerAction, PassAction)
from gym_tichu.envs.internals.utils import check_param, flatten, time_since

logger = logging.getLogger(__name__)


class Minimax(object):

    def __init__(self):
        self.maxdepth = None

    @property
    def info(self ) ->str:
        return self.__class__.__name__

    @timecall(immediate=False)
    def search(self, root_state: TichuState, max_depth: int, cheat: bool = True) -> PlayerAction:
        assert cheat, "cheat=False is not implemented"
        assert max_depth < 11  # Laptop Memory can't handle a depth of 11

        start_t = time()
        self.maxdepth = max_depth
        # possible actions
        asts = list(self.action_state_transisions(root_state))
        if len(asts) == 1:
            logger.debug("result of minimax: only one action; --> action:{}".format(asts[0][0]))
            return asts[0][0]

        # sort actions for better pruning
        asts_sorted = sorted(asts, key=lambda a_s: float("inf") if isinstance(a_s[0], PassAction) else len(a_s[0].combination))  # sort: low combinations first, Passing last.

        # start minimax search
        res = [(a, self.min_value(state=s.copy_discard_history(), alpha=-float("inf"), beta=float("inf"), depth=0, playerpos=root_state.player_pos)) for a, s in asts_sorted]
        action, val = max(res, key=itemgetter(1))
        logger.debug("Minimax search val: {}, time: {}".format(val, time_since(start_t)))
        return action

    def is_terminal(self, state, depth, playerpos)->bool:
        if depth >= self.maxdepth:
            return True
        else:
            return state.is_terminal()

    def action_state_transisions(self, state: TichuState)->Generator[Tuple[PlayerAction, TichuState], None, None]:
        yield from ((a, state.next_state(a)) for a in state.possible_actions_list)

    @timecall(immediate=False)
    def eval_state(self, state, playerpos)->float:
        if state.is_terminal():
            return self.terminal_heuristic(state, playerpos)
        else:
            return 14 - len(state.handcards[playerpos])  # TODO some better heuristic

    def terminal_heuristic(self, state, playerpos)->float:
        return (4 - state.ranking.index(playerpos))**3 if playerpos in state.ranking else -float('inf')   # TODO some better heuristic

    def max_value(self, state: TichuState, alpha: float, beta: float, depth: int, playerpos: int)->float:
        if self.is_terminal(state, depth, playerpos):
            # logging.debug("+max is terminal")
            return self.eval_state(state, playerpos)
        asts = self.action_state_transisions(state)
        v = -float("inf")
        for (a, s) in asts:
            v = max(v, self.min_value(s, alpha, beta, depth + 1, playerpos))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state: TichuState, alpha: float, beta: float, depth: int, playerpos: int)->float:
        if self.is_terminal(state, depth, playerpos):
            # logging.debug("-min is terminal")
            return self.eval_state(state, playerpos)
        asts = self.action_state_transisions(state)
        v = float("inf")
        for (a, s) in asts:
            v = min(v, self.max_value(s, alpha, beta, depth + 1, playerpos))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

