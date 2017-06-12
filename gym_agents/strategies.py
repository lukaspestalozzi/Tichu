"""
This module defines some strategies for subtasks of the tichu game.

- trading cards

- wishing

- announce tichus

- give dragon away
"""

import random
from typing import Callable, Optional, Collection, Tuple, Set

import logging
from gym_tichu.envs.internals import TichuState, CardRank, Card

TradingStrategyType     = Callable[[TichuState, int], Tuple[Card, Card, Card]]
WishStrategyType        = Callable[[TichuState, int], CardRank]
TichuStrategyType       = Callable[[TichuState, Set[int], int], bool]
DragonAwayStrategyType  = Callable[[TichuState, int], int]

logger = logging.getLogger(__name__)


def random_trading_strategy(state: TichuState, player: int) -> Tuple[Card, Card, Card]:
    sc = state.handcards[player].random_cards(3)
    return tuple(sc)


def random_wish_strategy(state: TichuState, player: int) -> CardRank:
    wish = random.choice([None, CardRank.TWO, CardRank.THREE, CardRank.FOUR, CardRank.FIVE, CardRank.SIX,
                          CardRank.SEVEN, CardRank.EIGHT, CardRank.NINE, CardRank.TEN, CardRank.J,
                          CardRank.Q, CardRank.K, CardRank.A])
    logger.debug("random wish strategy -> {}".format(wish))
    return wish


def never_announce_tichu_strategy(state: TichuState, already_announced: Set[int], player: int) -> bool:
    logger.debug("Never Tichu strategy -> False")
    return False


def always_announce_tichu_strategy(state: TichuState, already_announced: Set[int], player: int) -> bool:
    logger.debug("Always Tichu strategy -> True")
    return True


def make_random_tichu_strategy(announce_weight: float)->TichuStrategyType:
    assert 0.0 <= announce_weight <= 1.0
    return lambda state, already_announced, player: random.random() <= announce_weight


def give_dragon_to_the_right_strategy(state: TichuState, player: int) -> int:
    return (player + 1) % 4  # give dragon to player right


def give_dragon_to_the_left_strategy(state: TichuState, player: int) -> int:
    return (player - 1) % 4  # give dragon to player left