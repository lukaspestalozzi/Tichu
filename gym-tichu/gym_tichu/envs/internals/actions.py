
import abc
from collections import namedtuple
from typing import Optional, Generator, Sequence, List
from profilehooks import timecall

import itertools
import logging

from .utils import check_all_isinstance, check_isinstance
from .cards import Combination, DOG_COMBINATION, CardRank
from .error import NotSupportedError

__all__ = ("PlayerAction", "PlayCombination", "CardTrade", "TradeAction", "PlayFirst", "PlayDog", "PlayBomb",
           "PassAction", "TichuAction", "WinTrickAction", "GiveDragonAwayAction", "WishAction",
           "pass_actions", "tichu_actions", "no_tichu_actions", "play_dog_actions", "all_wish_actions_gen",
           "Trick", "FinishedTrick", "wishable_card_ranks")

CardTrade = namedtuple('CardTrade', ['from_', 'to', 'card'])

logger = logging.getLogger(__name__)


class PlayerAction(object, metaclass=abc.ABCMeta):

    def __init__(self, player_pos: int):
        assert player_pos in range(4)
        self._player_pos = player_pos

    @property
    def player_pos(self):
        return self._player_pos

    def __hash__(self)->int:
        return self._player_pos

    def __eq__(self, other: 'PlayerAction')->bool:
        return self.__class__ == other.__class__ and self.player_pos == other.player_pos

    def __repr__(self):
        return "{me.__class__.__name__}({me.player_pos})".format(me=self)


class PlayCombination(PlayerAction):
    __slots__ = ('_combination',)

    def __init__(self, player_pos: int, combination: Combination):
        super().__init__(player_pos=player_pos)
        self._combination = combination

    @property
    def combination(self):
        return self._combination

    def __hash__(self)->int:
        return hash((self._player_pos, self._combination))

    def __eq__(self, other: 'PlayCombination')->bool:
        return super().__eq__(other) and self.combination == other.combination

    def __repr__(self):
        return "{me.__class__.__name__}({me.player_pos}, {comb})".format(me=self, comb=repr(self._combination))

    def __str__(self):
        return "[{me.player_pos}, {me._combination}]".format(me=self)


class PlayFirst(PlayCombination):
    __slots__ = ()


class PlayDog(PlayFirst):
    __slots__ = ()

    def __init__(self, player_pos: int):
        super().__init__(player_pos=player_pos, combination=DOG_COMBINATION)

    def __str__(self):
        return "[{me.player_pos}, DOG]".format(me=self)


class PlayBomb(PlayCombination):
    __slots__ = ()

    def __init__(self, player_pos: int, combination: Combination):
        assert combination.is_bomb()
        super().__init__(player_pos=player_pos, combination=combination)


class PassAction(PlayerAction):
    __slots__ = ()

    def __str__(self):
        return "PASS({me.player_pos})".format(me=self)


class TichuAction(PlayerAction):
    __slots__ = ('_announce', '_grand')

    def __init__(self, player_pos: int, announce_tichu: bool, grand: bool=False):
        """
        
        :param player_pos: 
        :param announce_tichu: 
        :param grand: whether it is a grand tichu or not
        """
        super().__init__(player_pos=player_pos)
        self._announce = announce_tichu
        self._grand = grand

    @property
    def announce(self):
        return self._announce

    @property
    def grand(self):
        return self._grand

    def __hash__(self)->int:
        return hash((self._player_pos, self._announce))

    def __eq__(self, other: 'TichuAction')->bool:
        return super().__eq__(other) and self.announce == other.announce

    def __repr__(self):
        return "{me.__class__.__name__}({me.player_pos}, {me._announce})".format(me=self)

    def __str__(self):
        if self._announce:
            return "TICHU({me.player_pos})".format(me=self)
        else:
            return "NO_TICHU({me.player_pos})".format(me=self)


class WinTrickAction(PlayerAction):
    __slots__ = ('_trick',)

    def __init__(self, player_pos: int, trick: 'Trick'):
        super().__init__(player_pos=player_pos)
        self._trick = trick

    @property
    def trick(self):
        return self._trick

    def __hash__(self)->int:
        return hash((self._player_pos, self.trick))

    def __eq__(self, other: 'WinTrickAction')->bool:
        return super().__eq__(other) and self.trick == other.trick

    def __repr__(self):
        return "{me.__class__.__name__}({me.player_pos}, {tr})".format(me=self, tr=repr(self._trick))

    def __str__(self):
        return "{me.__class__.__name__}({me.player_pos}, points: {me.trick.points})".format(me=self)


class GiveDragonAwayAction(WinTrickAction):
    __slots__ = ('_to',)

    def __init__(self, player_from: int, player_to: int, trick: 'Trick'):
        assert player_to in range(4)
        super().__init__(player_pos=player_from, trick=trick)
        self._to = player_to

    @property
    def to(self):
        return self._to

    def __hash__(self)->int:
        return hash((self._player_pos, self._to, self.trick))

    def __eq__(self, other: 'GiveDragonAwayAction')->bool:
        return super().__eq__(other) and self.to == other.to

    def __repr__(self):
        return "{me.__class__.__name__}({me.player_pos} -> {me._to}, {tr})".format(me=self, tr=repr(self.trick))

    def __str__(self):
        return "{me.__class__.__name__}({me.player_pos} -> {me._to})".format(me=self)


class WishAction(PlayerAction):

    __slots__ = ("_cardval",)

    def __init__(self, player_pos, wish):
        assert wish is None or isinstance(wish, CardRank), wish
        super().__init__(player_pos=player_pos)
        self._wish = wish

    @property
    def wish(self):
        return self._wish

    def __hash__(self)->int:
        return hash((self._player_pos, self._wish))

    def __eq__(self, other: 'WishAction')->bool:
        return super().__eq__(other) and self.wish == other.wish

    def __repr__(self):
        return "{me.__class__.__name__}({me.player_pos}, {w})".format(me=self, w=repr(self._wish))

    def __str__(self):
        return "WISH({me.player_pos}, {me._wish})".format(me=self)


class TradeAction(PlayerAction, CardTrade):

    def __init__(self, from_, to, card):
        super().__init__(player_pos=from_)

    def __repr__(self):
        return "{me.__class__.__name__}({me.from_}: {crd} -> {me.to})".format(me=self, crd=repr(self.card))

    def __str__(self):
        return "Trade({me.from_}: {me.card} -> {me.to})".format(me=self)

# ###### PREDEFINED ACTIONS ######
# Pass
pass_actions = tuple((PassAction(k) for k in range(4)))

# Tichu
tichu_actions = tuple((TichuAction(k, True) for k in range(4)))
no_tichu_actions = tuple((TichuAction(k, False) for k in range(4)))

# Play Dog
play_dog_actions = tuple((PlayDog(k) for k in range(4)))

wishable_card_ranks = (CardRank.TWO, CardRank.THREE, CardRank.FOUR, CardRank.FIVE, CardRank.SIX,
                       CardRank.SEVEN, CardRank.EIGHT, CardRank.NINE, CardRank.TEN, CardRank.J,
                       CardRank.Q, CardRank.K, CardRank.A)


def all_wish_actions_gen(player_pos: int)->Generator:
    """
    :param player_pos: 
    :return: Generator yielding all wish actions with the given player id
    """
    yield WishAction(player_pos=player_pos, wish=None)
    for rank in wishable_card_ranks:
        yield WishAction(player_pos=player_pos, wish=rank)


# ######################### Trick #########################

class Trick(tuple):

    def __init__(self, actions_: Sequence[PlayerAction]=()):
        super().__init__()
        check_all_isinstance(self, PlayerAction)
        self._last_combination_action = 'NotInitialized'  # cache

    @property
    def last_combination(self)->Optional['Combination']:
        try:
            return self.last_combination_action.combination
        except AttributeError:
            return None

    @property
    def last_combination_action(self)->Optional['PlayCombination']:
        if self._last_combination_action == 'NotInitialized':
            try:
                self._last_combination_action = next((a for a in reversed(self) if isinstance(a, PlayCombination)))
            except StopIteration:
                self._last_combination_action = None
        return self._last_combination_action

    @property
    def last_action(self):
        return self[-1] if len(self) else None

    @property
    def points(self):
        return self.count_points()

    @property
    def winner(self):
        return self.last_combination_action.player_pos if len(self) else 'NoLeader'

    def combinations(self)->Generator[Combination, None, None]:
        yield from (act.combination for act in self if isinstance(act, PlayCombination))

    def count_points(self):
        return sum([comb.points for comb in self.combinations()])

    def is_empty(self):
        return len(self) == 0

    def is_dragon_trick(self):
        return self.last_combination.contains_dragon()

    def is_finished(self)->bool:
        return False

    def finish(self, last_action: Optional['PlayerAction']=None)->'FinishedTrick':
        """
        A Finished trick raises an Error when trying to add another action to it.
        :param last_action: if not None, then this action is appended to the finished trick and therefore is the last action of this trick
        :return: a new (finished) trick
        """
        acts = tuple(self)
        if last_action:
            acts += (last_action,)
        return FinishedTrick(acts)

    def __add__(self, action: 'PlayerAction')->'Trick':
        return Trick(itertools.chain(self, (action,)))

    def __str__(self):
        return "{me.__class__.__name__}[Leader: {me.winner}]: {tricks_str}".format(me=self, tricks_str=' -> '.join(map(str, iter(self))))

    def __repr__(self):
        return "{me.__class__.__name__}: {tricks_str}".format(me=self, tricks_str=' -> '.join(map(repr, iter(self))))


class FinishedTrick(Trick):
    """
    Special Trick, this trick is finished in the sense that no action can be added anymore.
    One player won this Trick already.
    """

    @property
    def winner(self):
        return self.last_combination_action.player_pos

    def finish(self, last_action=None):
        raise NotSupportedError("Can't finish a already finished Trick.")

    def is_finished(self)->bool:
        return True

    def __add__(self, *args, **kwargs):
        raise NotSupportedError("Can't add action to finished Trick.")


class MutableTrick(object):
    def __init__(self, actions_: Sequence[PlayerAction]=()):
        super().__init__()
        check_all_isinstance(actions_, PlayerAction)
        self._last_combination_action = 'NotInitialized'  # cache
        self._actions: List = list(actions_)

    @property
    def last_combination(self) -> Optional['Combination']:
        try:
            return self.last_combination_action.combination
        except AttributeError:
            return None

    @property
    def last_combination_action(self) -> Optional['PlayCombination']:
        if self._last_combination_action == 'NotInitialized':
            try:
                self._last_combination_action = next((a for a in reversed(self._actions) if isinstance(a, PlayCombination)))
            except StopIteration:
                self._last_combination_action = None
        return self._last_combination_action

    @property
    def last_action(self):
        return self._actions[-1] if len(self._actions) else None

    @property
    def points(self):
        return self.count_points()

    @property
    def winner(self):
        return self.last_combination_action.player_pos if len(self._actions) else 'NoLeader'

    @classmethod
    def from_immutable(cls, trick):
        return cls(trick)

    def combinations(self) -> Generator[Combination, None, None]:
        yield from (act.combination for act in self._actions if isinstance(act, PlayCombination))

    def count_points(self):
        return sum([comb.points for comb in self.combinations()])

    def is_empty(self):
        return len(self._actions) == 0

    def is_dragon_trick(self):
        return self.last_combination.contains_dragon()

    def is_finished(self) -> bool:
        return False

    def finish(self, last_action: Optional['PlayerAction'] = None) -> 'FinishedTrick':
        """
        A Finished trick raises an Error when trying to add another action to it.
        :param last_action: if not None, then this action is appended to the finished trick and therefore is the last action of this trick
        :return: a new (finished) trick
        """
        acts = tuple(self)
        if last_action:
            acts += (last_action,)
        return FinishedTrick(acts)

    def append(self, action: 'PlayerAction'):
        check_isinstance(action, PlayerAction)
        if isinstance(action, PlayCombination):
            self._last_combination_action = action
        self._actions.append(action)

    def __iter__(self):
        return self._actions.__iter__()

    def __len__(self):
        return len(self._actions)

    def __repr__(self):
        return "{me.__class__.__name__}[Leader: {me.winner}]: {tricks_str}".format(me=self, tricks_str=' -> '.join(map(str, iter(self._actions))))


