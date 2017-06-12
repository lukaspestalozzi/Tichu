import abc
import itertools
import random as rnd
import logging

from enum import Enum
from collections import defaultdict
from typing import Union, Optional, Sequence, Set, Collection, Iterable, List, Tuple, Dict, Generator, Any
from profilehooks import timecall

from .utils import check_true, check_param, check_isinstance, check_all_isinstance, ignored, TypedFrozenSet, TypedTuple, indent
from .error import LogicError


__author__ = 'Lukas Pestalozzi'

__all__ = ('CardRank', 'CardSuit', 'Card',
           'CardSet', 'Deck',
           'Combination', 'Single', 'DOG_COMBINATION', 'DRAGON_COMBINATION',
           'Pair', 'Trio', 'FullHouse', 'PairSteps', 'Straight',
           'SquareBomb', 'StraightBomb',
           'GeneralCombination', 'all_general_combinations_gen')

logger = logging.getLogger(__name__)

# ######################### CARD #########################


class ComparableEnum(Enum):
    """
    Enum that allows comparing instances with >, >=, <=, <
    """

    # functions to compare the enum
    def _raise_cant_compare(self, other):
        raise TypeError(
                "operation not supported between instances of {} and {}.".format(self.__class__, other.__class__))

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        self._raise_cant_compare(other)

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        self._raise_cant_compare(other)

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        self._raise_cant_compare(other)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        self._raise_cant_compare(other)


class CardRank(ComparableEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    J = 11
    Q = 12
    K = 13
    A = 14
    DRAGON = 15
    PHOENIX = 1.5
    MAHJONG = 1
    DOG = 0

    def __init__(self, _):
        if self.value == 5:
            self._points = 5
        elif self.value in [10, 13]:
            self._points = 10
        elif self.value == 15:
            self._points = 25
        elif self.value == 1.5:
            self._points = -25
        else:
            self._points = 0
        u"\U0001F426"
        self._repr = "CardRank({} {})".format(self.name, self.value)
        # DRAGON
        if self.value == 15:
            self._str = "DRAGON"  # u"\U0001F409"  # dragon
        # PHOENIX
        elif self.value == 1.5:
            self._str = "PH"  # u"\U0001F010"  # (U+1F010)
        # MAHJONG
        elif self.value == 1:
            self._str = "1"  # u"\U0001F004"  # 'MAHJONG TILE ONE OF BAMBOOS'
        # DOG
        elif self.value == 0:
            self._str = "DOG"  # u"\U0001F436"  # dog
        # J - A
        elif self.value > 10:
            self._str = str(self.name)
        # 2 - 10
        else:
            self._str = str(self.value)

    @property
    def height(self)->int:
        return self.value

    @property
    def points(self)->int:
        return self._points

    @staticmethod
    def from_name(name)->'CardRank':
        """
        >>> CardRank.from_name('TWO').name
        'TWO'
        >>> CardRank.from_name('THREE').name
        'THREE'
        >>> CardRank.from_name('FOUR').name
        'FOUR'
        >>> CardRank.from_name('FIVE').name
        'FIVE'
        >>> CardRank.from_name('SIX').name
        'SIX'
        >>> CardRank.from_name('SEVEN').name
        'SEVEN'
        >>> CardRank.from_name('EIGHT').name
        'EIGHT'
        >>> CardRank.from_name('NINE').name
        'NINE'
        >>> CardRank.from_name('TEN').name
        'TEN'
        >>> CardRank.from_name('J').name
        'J'
        >>> CardRank.from_name('Q').name
        'Q'
        >>> CardRank.from_name('K').name
        'K'
        >>> CardRank.from_name('A').name
        'A'
        >>> CardRank.from_name('DOG').name
        'DOG'
        >>> CardRank.from_name('PHOENIX').name
        'PHOENIX'
        >>> CardRank.from_name('DRAGON').name
        'DRAGON'
        >>> CardRank.from_name('MAHJONG').name
        'MAHJONG'
        >>> CardRank.from_name('Not existant')
        Traceback (most recent call last):
        ...
        ValueError: There is no CardRank with name 'Not existant'. possible names: ['A', 'DOG', 'DRAGON', 'EIGHT', 'FIVE', 'FOUR', 'J', 'K', 'MAHJONG', 'NINE', 'PHOENIX', 'Q', 'SEVEN', 'SIX', 'TEN', 'THREE', 'TWO']

        :param name:
        :return:
        """
        try:
            return next(cv for cv in CardRank if cv.name==name)
        except KeyError:
            raise ValueError(f"There is no CardRank with name '{name}'. possible names: {sorted(cv.name for cv in CardRank)}")

    @staticmethod
    def from_value(val):
        try:
            return next(cv for cv in CardRank if cv.value==val)
        except StopIteration:
            raise ValueError(f"There is no CardRank with value '{val}'. possible names: {sorted(cv.value for cv in CardRank)}")

    def __add__(self, other: Union[int, 'CardRank'])->int:
        i = other.value if isinstance(other, CardRank) else other
        return self.value + i

    def __sub__(self, other: Union[int, 'CardRank'])->int:
        i = other.value if isinstance(other, CardRank) else other
        return self.value - i

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.name, self.height)

    def __str__(self):
        return self._str


class CardSuit(ComparableEnum):
    # TODO Fix unicode
    SWORD = u"\u2694"  # 'CROSSED SWORDS' (U+2694)
    JADE = u'\u2666'  # 'Black Diamond Suit' U+2666
    PAGODA = u'\u2665'
    STAR = u"\u2605"  # 'BLACK STAR' (U+2605)
    SPECIAL = u'\u1f0cf'

    def __init__(self, unicode):
        self._unicode = unicode
        self._shortname = self._name_[:2]
        self._repr = "{}({}, {})".format(self.__class__.__name__, self.name, self._unicode)
        self._nbr = (0 if self._name_ == 'SWORD' else
                     1 if self._name_ == 'JADE' else
                     2 if self._name_ == 'PAGODA' else
                     3 if self._name_ == 'HOUSE' else
                     4 if self._name_ == 'SPECIAL' else
                     None)

    @property
    def unicode(self):
        return self._unicode

    @property
    def shortname(self):
        return self._shortname

    @property
    def number(self):
        """
        SWORD:   0
        JADE:    1
        PAGODA:  2
        HOUSE:   3
        SPECIAL: 4

        :return: The number for this suit
        """
        return self._nbr

    def __unicode__(self):
        return self._unicode

    def pretty_string(self):
        return self._unicode

    def __repr__(self):
        return self._repr

    def __str__(self):
        return self.unicode


class Card(ComparableEnum):
    DOG = (CardRank.DOG, CardSuit.SPECIAL, 0)
    MAHJONG = (CardRank.MAHJONG, CardSuit.SPECIAL, 1)
    DRAGON = (CardRank.DRAGON, CardSuit.SPECIAL, 2)
    PHOENIX = (CardRank.PHOENIX, CardSuit.SPECIAL, 3)

    TWO_JADE = (CardRank.TWO, CardSuit.JADE, 4)
    THREE_JADE = (CardRank.THREE, CardSuit.JADE, 5)
    FOUR_JADE = (CardRank.FOUR, CardSuit.JADE, 6)
    FIVE_JADE = (CardRank.FIVE, CardSuit.JADE, 7)
    SIX_JADE = (CardRank.SIX, CardSuit.JADE, 8)
    SEVEN_JADE = (CardRank.SEVEN, CardSuit.JADE, 9)
    EIGHT_JADE = (CardRank.EIGHT, CardSuit.JADE, 10)
    NINE_JADE = (CardRank.NINE, CardSuit.JADE, 11)
    TEN_JADE = (CardRank.TEN, CardSuit.JADE, 12)
    J_JADE = (CardRank.J, CardSuit.JADE, 13)
    Q_JADE = (CardRank.Q, CardSuit.JADE, 14)
    K_JADE = (CardRank.K, CardSuit.JADE, 15)
    A_JADE = (CardRank.A, CardSuit.JADE, 16)

    TWO_HOUSE = (CardRank.TWO, CardSuit.STAR, 17)
    THREE_HOUSE = (CardRank.THREE, CardSuit.STAR, 18)
    FOUR_HOUSE = (CardRank.FOUR, CardSuit.STAR, 19)
    FIVE_HOUSE = (CardRank.FIVE, CardSuit.STAR, 20)
    SIX_HOUSE = (CardRank.SIX, CardSuit.STAR, 21)
    SEVEN_HOUSE = (CardRank.SEVEN, CardSuit.STAR, 22)
    EIGHT_HOUSE = (CardRank.EIGHT, CardSuit.STAR, 23)
    NINE_HOUSE = (CardRank.NINE, CardSuit.STAR, 24)
    TEN_HOUSE = (CardRank.TEN, CardSuit.STAR, 25)
    J_HOUSE = (CardRank.J, CardSuit.STAR, 26)
    Q_HOUSE = (CardRank.Q, CardSuit.STAR, 27)
    K_HOUSE = (CardRank.K, CardSuit.STAR, 28)
    A_HOUSE = (CardRank.A, CardSuit.STAR, 29)

    TWO_SWORD = (CardRank.TWO, CardSuit.SWORD, 30)
    THREE_SWORD = (CardRank.THREE, CardSuit.SWORD, 31)
    FOUR_SWORD = (CardRank.FOUR, CardSuit.SWORD, 32)
    FIVE_SWORD = (CardRank.FIVE, CardSuit.SWORD, 33)
    SIX_SWORD = (CardRank.SIX, CardSuit.SWORD, 34)
    SEVEN_SWORD = (CardRank.SEVEN, CardSuit.SWORD, 35)
    EIGHT_SWORD = (CardRank.EIGHT, CardSuit.SWORD, 36)
    NINE_SWORD = (CardRank.NINE, CardSuit.SWORD, 37)
    TEN_SWORD = (CardRank.TEN, CardSuit.SWORD, 38)
    J_SWORD = (CardRank.J, CardSuit.SWORD, 39)
    Q_SWORD = (CardRank.Q, CardSuit.SWORD, 40)
    K_SWORD = (CardRank.K, CardSuit.SWORD, 41)
    A_SWORD = (CardRank.A, CardSuit.SWORD, 42)

    TWO_PAGODA = (CardRank.TWO, CardSuit.PAGODA, 43)
    THREE_PAGODA = (CardRank.THREE, CardSuit.PAGODA, 44)
    FOUR_PAGODA = (CardRank.FOUR, CardSuit.PAGODA, 45)
    FIVE_PAGODA = (CardRank.FIVE, CardSuit.PAGODA, 46)
    SIX_PAGODA = (CardRank.SIX, CardSuit.PAGODA, 47)
    SEVEN_PAGODA = (CardRank.SEVEN, CardSuit.PAGODA, 48)
    EIGHT_PAGODA = (CardRank.EIGHT, CardSuit.PAGODA, 49)
    NINE_PAGODA = (CardRank.NINE, CardSuit.PAGODA, 50)
    TEN_PAGODA = (CardRank.TEN, CardSuit.PAGODA, 51)
    J_PAGODA = (CardRank.J, CardSuit.PAGODA, 52)
    Q_PAGODA = (CardRank.Q, CardSuit.PAGODA, 53)
    K_PAGODA = (CardRank.K, CardSuit.PAGODA, 54)
    A_PAGODA = (CardRank.A, CardSuit.PAGODA, 55)

    def __init__(self, cardrank: CardRank, cardsuit: CardSuit, number: int):
        self._suit = cardsuit
        self._cardrank = cardrank
        self._nbr = number

        # precompute strings and hash
        self._hash = hash((cardrank, cardsuit))

        self._str = (f"{str(self._cardrank)}{self._suit.pretty_string()}" if self._suit is not CardSuit.SPECIAL
                     else f"{str(self._cardrank)}")

    @property
    def suit(self)->CardSuit:
        return self._suit

    @property
    def card_rank(self)->CardRank:
        return self._cardrank

    @property
    def rank(self) -> CardRank:
        return self._cardrank

    @property
    def card_height(self):
        return self._cardrank.height

    @property
    def points(self):
        return self._cardrank.points

    @property
    def number(self):
        """
        The cards are numerated from 0 to 55.

        :return: The unique number for this card.
        """
        return self._nbr

    def __eq__(self, other):  # TODO question, raise error when classes not the same?
        return self.__class__ is other.__class__ and self.card_rank == other.card_rank and self._suit == other.suit

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return self._str

    def __hash__(self):
        return self._hash

# Dict mapping a card-rank value to the card with that rank and the sword suit
card_rank_to_sword_card = {
    2: Card.TWO_SWORD,
    3: Card.THREE_SWORD,
    4: Card.FOUR_SWORD,
    5: Card.FIVE_SWORD,
    6: Card.SIX_SWORD,
    7: Card.SEVEN_SWORD,
    8: Card.EIGHT_SWORD,
    9: Card.NINE_SWORD,
    10: Card.TEN_SWORD,
    11: Card.J_SWORD,
    12: Card.Q_SWORD,
    13: Card.K_SWORD,
    14: Card.A_SWORD,
}


# ######################### CARD SET #########################

class CardSet(TypedFrozenSet):

    __slots__ = ('_rank_dict', '_all_generalcombs_cache', '_poss_combs_cache', '_singles_cache', '_pairs_cache',
                 '_trios_cache', '_straights_cache', '_pairsteps_cache', '_fullhouses_cache', '_straightbombs_cache')

    def __init__(self, cards: Iterable=()):
        """
        :param cards: An iterable containing Card instances. If it is another CardSet, then it will be copied.
        """
        cards = set(cards)  # if cards is a generator
        check_all_isinstance(cards, Card)
        super().__init__(cards)
        # caches
        self._rank_dict = None
        self._all_generalcombs_cache = None

        self._poss_combs_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['Combination']] = dict()

        self._singles_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['Single']] = dict()
        self._pairs_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['Pair']] = dict()
        self._trios_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['Trio']] = dict()
        # self._quads_cache = None  # probably Not worth it
        self._straights_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['Straight']] = dict()
        self._pairsteps_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['PairSteps']] = dict()
        self._fullhouses_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['FullHouse']] = dict()
        self._straightbombs_cache: Dict[Tuple[Optional['Combination'], Optional[CardRank]], List['StraightBomb']] = dict()

    def rank_dict(self, exclude_special: bool=False)->Dict[CardRank, List[Card]]:
        """
        :return: a dict mapping the card_ranks appearing in self._cards to the list of corresponding cards.
        """
        # create dict if not already created
        if self._rank_dict is None:
            temp_dict = defaultdict(lambda: [])
            for c in self:
                temp_dict[c.card_rank].append(c)
            self._rank_dict = {k: tuple(sorted(v)) for k, v in temp_dict.items()}

        # return the right dict
        if exclude_special:
            # TODO speed, maybe faster?
            return {k: cards for k, cards in self._rank_dict.items() if cards[0].suit is not CardSuit.SPECIAL}
        else:
            return dict(self._rank_dict)

    def count_points(self)->int:
        """
        :return the Tichu points in this set of cards.
        """
        return sum([c.points for c in self])

    def sorted_tuple(self, *args, **kwargs):
        """
        :param args, kwargs: same parameters as for the built in 'sorted' method
        :return: The elements as a sorted tuple
        """
        return tuple(sorted(self, *args, **kwargs))

    def random_cards(self, n=1):
        """
        :param n: int > 0
        :return: n random cards.
        """
        cds = list(self)
        rnd.shuffle(cds)
        return cds[:n]

    def contains_cardrank(self, rank: CardRank):
        return rank in self.rank_dict()

    # Combinations methods
    def possible_combinations(self, played_on: Optional['Combination']=None, contains_rank: Optional[CardRank]=None)->Generator['Combination', None, None]:
        """
        Note, caches the result
        
        :param played_on: 
        :param contains_rank: 
        :return: An Generator yielding all possible combinations respecting played_on and contains_rank
        """
        # TODO make better adapted to different parametervalues (eg. try to fethc (None, None) and adapt)
        # logger.debug("possible_combinations with arguments {} and {}".format(played_on, contains_rank))
        param_tup = (played_on, contains_rank)
        if param_tup not in self._poss_combs_cache:
            # logger.debug("Not cached")
            self._poss_combs_cache[param_tup] = list(self._possible_combinations_gen(played_on=played_on, contains_rank=contains_rank))
        # logger.debug("yield from : {}".format(self._poss_combs_cache[param_tup]))
        yield from self._poss_combs_cache[param_tup]

    def _possible_combinations_gen(self, played_on: Optional['Combination']=None, contains_rank: Optional[CardRank]=None)->Generator['Combination', None, None]:
        assert played_on is None or isinstance(played_on, Combination)
        assert contains_rank is None or isinstance(contains_rank, CardRank)

        # TODO remove the played_on from single, pair, trio squarebomb, or make it more efficient

        if played_on is None:
            yield from itertools.chain(
                    self.singles(played_on=played_on, contains_rank=contains_rank),
                    self.all_bombs(played_on=played_on, contains_rank=contains_rank),
                    self.pairs(played_on=played_on, contains_rank=contains_rank),
                    self.trios(played_on=played_on, contains_rank=contains_rank),
                    self.straights(played_on=played_on, contains_rank=contains_rank),
                    self.fullhouses(played_on=played_on, contains_rank=contains_rank),
                    self.pairsteps(played_on=played_on, contains_rank=contains_rank)
                )
        else:
            if Card.DOG in played_on:
                assert len(played_on) == 1
                return   # it is not possible to play on the dog

            if isinstance(played_on, Bomb):
                yield from self.all_bombs(played_on=played_on, contains_rank=contains_rank)  # only higher bombs
            else:
                yield from self.all_bombs(contains_rank=contains_rank)  # all bombs

            if Card.DRAGON in played_on:
                assert len(played_on) == 1
                return  # only bombs can beat the Dragon

            elif isinstance(played_on, Single):
                # all single cards higher than the played_on
                yield from self.singles(played_on=played_on, contains_rank=contains_rank)

            elif isinstance(played_on, Pair):
                # all pairs higher than the played_on.any_card
                yield from self.pairs(played_on=played_on, contains_rank=contains_rank)

            elif isinstance(played_on, Trio):
                # all trios higher than the played_on.any_card
                yield from self.trios(played_on=played_on, contains_rank=contains_rank)

            elif isinstance(played_on, PairSteps):
                # all higher pairsteps
                yield from self.pairsteps(played_on=played_on, contains_rank=contains_rank)

            elif isinstance(played_on, Straight):
                # all higher straights
                yield from self.straights(played_on=played_on, contains_rank=contains_rank)

    def singles(self, played_on: Optional['Single']=None, contains_rank: Optional[CardRank]=None)->Generator['Single', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._singles_cache:
            self._singles_cache[param_tup] = list(self._singles_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._singles_cache[param_tup]

    def _singles_gen(self, played_on: Optional['Single']=None, contains_rank: Optional[CardRank]=None)->Generator['Single', None, None]:
        # TODO handle presence of streetbomb

        if played_on is not None and not isinstance(played_on, Single):
            return  # Nothing to do here

        rank_dict = self.rank_dict()
        if contains_rank:
            try:
                single = Single(rank_dict[contains_rank][0])
            except KeyError:
                pass  # don't have that rank
            else:
                if single.can_be_played_on(played_on):
                    yield single
            return  # no more possible

        # contains_rank = None
        if played_on:
            singles = (Single(crds[0]) for crds in rank_dict.values())
            yield from (s for s in singles if s.can_be_played_on(played_on))
        else:
            # All singles
            yield from (Single(crds[0]) for crds in rank_dict.values())

    def pairs(self, played_on: Optional['Pair']=None, contains_rank: Optional[CardRank]=None)->Generator['Pair', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._pairs_cache:
            self._pairs_cache[param_tup] = list(self._pairs_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._pairs_cache[param_tup]

    def _pairs_gen(self, played_on: Optional['Pair']=None, contains_rank: Optional[CardRank]=None)->Generator['Pair', None, None]:
        # TODO handle presence of streetbomb
        # TODO cache
        if played_on is not None and not isinstance(played_on, Pair):
            return  # Nothing to do here

        rank_dict = self.rank_dict(exclude_special=True)
        if contains_rank:
            rank_dict = {contains_rank: rank_dict[contains_rank]} if contains_rank in rank_dict else dict()

        # phoenix
        if Card.PHOENIX in self:
            yield from (pair for pair in (Pair(cards[0], Card.PHOENIX) for cards in rank_dict.values()) if pair.can_be_played_on(played_on))

        # normal pairs
        for crds in rank_dict.values():
            if len(crds) >= 2:
                pair = Pair(crds[0], crds[1])
                if pair.can_be_played_on(played_on):
                    yield pair

    def trios(self, played_on: Optional['Trio']=None, contains_rank: Optional[CardRank]=None)->Generator['Trio', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._trios_cache:
            self._trios_cache[param_tup] = list(self._trios_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._trios_cache[param_tup]

    def _trios_gen(self, played_on: Optional['Trio']=None, contains_rank: Optional[CardRank]=None)->Generator['Trio', None, None]:
        # TODO handle presence of streetbomb
        # TODO cache
        if played_on is not None and not isinstance(played_on, Trio):
            return  # Nothing to do here

        rank_dict = self.rank_dict(exclude_special=True)
        if contains_rank:
            rank_dict = {contains_rank: rank_dict[contains_rank]} if contains_rank in rank_dict else dict()

        # phoenix
        have_phoenix = Card.PHOENIX in self
        for crds in rank_dict.values():
            if have_phoenix and len(crds) >= 2:
                trio = Trio(crds[0], crds[1], Card.PHOENIX)
                if trio.can_be_played_on(played_on):
                    yield trio

            if len(crds) >= 3:
                trio = Trio(crds[0], crds[1], crds[2])
                if trio.can_be_played_on(played_on):
                    yield trio

    def squarebombs(self, played_on: Optional['Combination']=None, contains_rank: Optional[CardRank]=None)->Generator['SquareBomb', None, None]:
        # TODO cache
        if isinstance(played_on, StraightBomb):
            return  # Nothing to do here

        rank_dict = self.rank_dict()
        if contains_rank:
            rank_dict = {contains_rank: rank_dict[contains_rank]} if contains_rank in rank_dict else dict()

        for cards in rank_dict.values():
            if len(cards) == 4:
                sq = SquareBomb(*cards)
                if sq.can_be_played_on(played_on):
                    yield sq

    def straightbombs(self, played_on: Optional['Combination']=None, contains_rank: Optional[CardRank]=None)->Generator['StraightBomb', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._straightbombs_cache:
            self._straightbombs_cache[param_tup] = list(self._straightbombs_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._straightbombs_cache[param_tup]

    def _straightbombs_gen(self, played_on: Optional['Combination']=None, contains_rank: Optional[CardRank]=None)->Generator['StraightBomb', None, None]:
        # group by card suit
        suitdict = defaultdict(list)
        for c in self:
            suitdict[c.suit].append(c)

        # look only at cards of same suit
        for suit, cards in suitdict.items():
            if len(cards) >= 5:  # must be at least 5 to be a straight (also excludes special cards)
                sbombs = (StraightBomb(st) for st in CardSet(cards).straights(contains_rank=contains_rank))
                for sbomb in sbombs:
                    if sbomb.can_be_played_on(played_on):
                        yield sbomb

    def all_bombs(self, played_on: Optional['Combination']=None, contains_rank: Optional[CardRank]=None)->Generator['Bomb', None, None]:
        return itertools.chain(self.squarebombs(played_on=played_on, contains_rank=contains_rank),
                               self.straightbombs(played_on=played_on, contains_rank=contains_rank))

    def straights(self, played_on: Optional['Straight'] = None, contains_rank: Optional[CardRank] = None)->Generator['Straight', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._straights_cache:
            self._straights_cache[param_tup] = list(self._straights_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._straights_cache[param_tup]

    def _straights_gen(self, played_on: Optional['Straight'] = None, contains_rank: Optional[CardRank] = None)->Generator['Straight', None, None]:
        if played_on and not isinstance(played_on, Straight):
            return  # Nothing to do here

        # TODO use played_on better

        has_phoenix = Card.PHOENIX in self

        if len(self) < (5 if played_on is None else len(played_on)):
            # if not enough cards are available -> return.
            return

        elif contains_rank and not self.contains_cardrank(contains_rank):
            # does not contain the 'contains_value' card -> return
            return
        else:
            # TODO speed
            sorted_cards = sorted(filter(lambda card: card is not Card.PHOENIX and card is not Card.DOG and card is not Card.DRAGON, self))
            # print("sorted_cards", sorted_cards)

            # TODO to exclude 'logical' duplicates, remove all duplicated ranks from sorted_cards

            next_card: Dict[int, List[Card]] = defaultdict(lambda: [])  # card rank -> list of cards with rank 1 higher
            for c in sorted_cards:
                next_card[c.rank.value - 1].append(c)

            def gen_from(card, remlength, ph):
                if remlength <= 1:
                    yield {card: card}  # finish a straight with this card

                # a straight for one possible continuation
                next_cards = next_card[card.rank.value]
                if len(next_cards) > 0:
                    for st in gen_from(next_cards[0], remlength - 1, ph=ph):
                        yield {card: card, **st}

                # Phoenix:
                if ph is None and has_phoenix:
                    # finish the straight with the Phoenix:
                    if remlength <= 2 and card.rank is not CardRank.A:
                        phoenix_as = card_rank_to_sword_card[card.rank + 1]
                        yield {card: card, Card.PHOENIX: phoenix_as}

                    # take phoenix instead of card
                    if card is not Card.MAHJONG:
                        if len(next_cards) > 0:
                            for st in gen_from(next_cards[0], remlength - 1, ph=card):
                                yield {Card.PHOENIX: card, **st}

                    # take phoenix to jump a value
                    if card.rank < CardRank.K and len(next_card[card.rank.value]) == 0:  # can not jump the As, and only jump if there is no next card
                        after_next_cards = next_card[card.rank.value + 1]
                        if len(after_next_cards) > 0:  # there is a card to 'land'
                            phoenix_as = card_rank_to_sword_card[card.rank + 1]
                            for st in gen_from(after_next_cards[0], remlength - 2, ph=phoenix_as):
                                yield {card: card, Card.PHOENIX: phoenix_as, **st}

            def gen_all_straights():
                """ Take all possible starting cards and generate straights from them """
                max_start_rank = CardRank.TEN  # there is no possible straight starting from J (must have length 5)
                if contains_rank:
                    max_start_rank = min(max_start_rank, contains_rank)  # straight starting from a higher value than contains_rank, can not contain that rank

                for c in sorted_cards:
                    if c.rank <= max_start_rank:
                        yield from gen_from(c, 5, ph=None)  # all straights starting with normal card
                        # all straights starting with the Phoenix:
                        if has_phoenix and c.rank > CardRank.TWO:
                            phoenix = card_rank_to_sword_card[c.rank - 1]
                            for st in gen_from(c, 4, ph=phoenix):
                                yield {Card.PHOENIX: phoenix, **st}

            # make and yield the Straights:
            gen = (Straight(set(st.keys()), phoenix_as=st.get(Card.PHOENIX, None)) for st in gen_all_straights())
            if contains_rank:
                yield from (st for st in gen if st.contains_cardrank(contains_rank) and st.can_be_played_on(played_on))
            else:
                yield from (st for st in gen if st.can_be_played_on(played_on))

    def fullhouses(self, played_on: Optional['FullHouse'] = None, contains_rank: Optional[CardRank] = None)->Generator['FullHouse', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._fullhouses_cache:
            self._fullhouses_cache[param_tup] = list(self._fullhouses_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._fullhouses_cache[param_tup]

    def _fullhouses_gen(self, played_on: Optional['FullHouse'] = None, contains_rank: Optional[CardRank] = None)->Generator['FullHouse', None, None]:
        if played_on is not None and not isinstance(played_on, FullHouse):
            return  # Nothing to do here

        trios = list(self.trios(played_on=played_on.trio if played_on else None))
        pairs = list(self.pairs())
        if contains_rank:
            for t in trios:
                t_contains = t.contains_cardrank(contains_rank)
                for p in pairs:
                    if t_contains or p.contains_cardrank(contains_rank):
                        with ignored(TypeError, ValueError):
                            fh = FullHouse(pair=p, trio=t)
                            yield fh
        else:
            for t in trios:
                for p in pairs:
                    with ignored(TypeError, ValueError):
                        fh = FullHouse(pair=p, trio=t)
                        yield fh

    def pairsteps(self, played_on: Optional['PairSteps'] = None, contains_rank: Optional[CardRank] = None)->Generator['PairSteps', None, None]:
        param_tup = (played_on, contains_rank)
        if param_tup not in self._pairsteps_cache:
            self._pairsteps_cache[param_tup] = list(self._pairsteps_gen(played_on=played_on, contains_rank=contains_rank))
        yield from self._pairsteps_cache[param_tup]

    def _pairsteps_gen(self, played_on: Optional['PairSteps'] = None, contains_rank: Optional[CardRank] = None)->Generator['PairSteps', None, None]:
        if played_on is not None and not isinstance(played_on, PairSteps) or len(self) < 4:
            return  # Nothing to do here

        sorted_pairs = sorted(self.pairs(), key=lambda p: p.height)
        next_pair_no_ph = defaultdict(lambda: [])
        next_pair_with_ph = defaultdict(lambda: [])
        for p in sorted_pairs:
            if p.contains_phoenix():
                next_pair_with_ph[p.height-1].append(p)
            else:
                next_pair_no_ph[p.height - 1].append(p)

        def gen_from(pair, remlength, ph_used)->Generator[List['Pair'], None, None]:
            if remlength <= 1:
                yield [pair]

            # continue without phoenix:
            with ignored(StopIteration, IndexError):
                for ps in gen_from(next_pair_no_ph[pair.height][0], remlength - 1, ph_used=ph_used):
                    yield [pair] + ps

            # continue with phoenix:
            if not ph_used:
                with ignored(StopIteration, IndexError):
                    for ps in gen_from(next_pair_with_ph[pair.height][0], remlength - 1, ph_used=True):
                        yield [pair] + ps

        def gen_all_pairsteps()->Generator[List['Pair'], None, None]:
            """ Take all possible starting pairs and generate pairsteps from them """
            max_height = CardRank.A.value  # there is no possible pairstep starting from As (must have length 2)
            if contains_rank:
                max_height = min(max_height, contains_rank.value)  # straight starting from a higher value than contains_val, can not contain that val

            for pair in sorted_pairs:
                if pair.height <= max_height:
                    yield from gen_from(pair, 2, ph_used=pair.contains_phoenix())  # all steps starting with the pair

        # make and yield the pairsteps:
        gen = (PairSteps(pairs) for pairs in gen_all_pairsteps())
        if contains_rank:
            yield from (ps for ps in gen if ps.contains_cardrank(contains_rank) and ps.can_be_played_on(played_on))
        else:
            yield from (ps for ps in gen if ps.can_be_played_on(played_on))

    def all_general_combinations(self)->Set['GeneralCombination']:
        if self._all_generalcombs_cache is None:
            retset = set()

            rank_dict = self.rank_dict(exclude_special=False)
            # ####### singles, pairs, trios, squarebombs ##########
            pairs = set()
            trios = set()
            for rank, cards in rank_dict.items():
                if len(cards) >= 1:
                    retset.add(GeneralCombination.from_combination(Single(cards[0])))
                if len(cards) >= 2:
                    pair = Pair(*cards[:2])
                    pairs.add(pair)
                    retset.add(GeneralCombination.from_combination(pair))
                if len(cards) >= 3:
                    trio = Trio(*cards[:3])
                    trios.add(trio)
                    retset.add(GeneralCombination.from_combination(trio))
                if len(cards) == 4:
                    retset.add(GeneralCombination.from_combination(SquareBomb(*cards)))

            # ####### FullHouses ##########
            if len(pairs):
                for trio in trios:
                    for pair in pairs:
                        try:
                            retset.add(GeneralCombination.from_combination(FullHouse(pair=pair, trio=trio)))
                            break  # only need 1 pair that works
                        except ValueError:  # for example this does not work: (PAIR(Q★,Q♥), TRIO(Q★,Q♥,Q⚔)), so have to try next pair to make a valid Fullhouse
                            pass

            # ####### STRAIGHTS ##########
            for st in self.straights():
                retset.add(GeneralCombination.from_combination(st))

            # ####### SttaightBomb ##########
            for stb in self.straightbombs():
                retset.add(GeneralCombination.from_combination(stb))

            # ####### PAIRSTEPS ##########
            for ps in self.pairsteps():
                retset.add(GeneralCombination.from_combination(ps))

            self._all_generalcombs_cache = frozenset(retset)

        return self._all_generalcombs_cache

    def __str__(self):
        return "{{{}}}".format(', '.join(map(str, sorted(self))))

    def __repr__(self):
        return "{}{{{}}}".format(self.__class__.__name__, ', '.join(map(repr, sorted(self))))


class Deck(list):
    """
    Deck is just a list of cards.
    """

    def __init__(self, full: bool=True, cards: Iterable[Card]=list()):
        """
        :param full: if True, the argument cards is ignored and a full deck is created. Default is True
        :param cards: The cards initially in the Deck. Ignored when 'full=True'
        """
        if full:
            cards_to_add = [
                Card.PHOENIX, Card.DRAGON, Card.MAHJONG, Card.DOG,
                Card.TWO_JADE, Card.THREE_JADE, Card.FOUR_JADE, Card.FIVE_JADE, Card.SIX_JADE, Card.SEVEN_JADE,
                Card.EIGHT_JADE, Card.NINE_JADE, Card.TEN_JADE, Card.J_JADE, Card.Q_JADE, Card.K_JADE, Card.A_JADE,
                Card.TWO_HOUSE, Card.THREE_HOUSE, Card.FOUR_HOUSE, Card.FIVE_HOUSE, Card.SIX_HOUSE, Card.SEVEN_HOUSE,
                Card.EIGHT_HOUSE, Card.NINE_HOUSE, Card.TEN_HOUSE, Card.J_HOUSE, Card.Q_HOUSE, Card.K_HOUSE,
                Card.A_HOUSE,
                Card.TWO_SWORD, Card.THREE_SWORD, Card.FOUR_SWORD, Card.FIVE_SWORD, Card.SIX_SWORD, Card.SEVEN_SWORD,
                Card.EIGHT_SWORD, Card.NINE_SWORD, Card.TEN_SWORD, Card.J_SWORD, Card.Q_SWORD, Card.K_SWORD,
                Card.A_SWORD,
                Card.TWO_PAGODA, Card.THREE_PAGODA, Card.FOUR_PAGODA, Card.FIVE_PAGODA, Card.SIX_PAGODA,
                Card.SEVEN_PAGODA, Card.EIGHT_PAGODA, Card.NINE_PAGODA, Card.TEN_PAGODA, Card.J_PAGODA, Card.Q_PAGODA,
                Card.K_PAGODA, Card.A_PAGODA
            ]
        else:
            cards_to_add = set(cards)

        super().__init__(cards_to_add)
        assert len(self) == len(cards_to_add)

    def split(self, nbr_piles=4, random_=True)->List[List[Card]]:
        """
        :param nbr_piles: Splits the deck into 'nbr_piles' same sized piles (defualt is 4).
        The size of the deck must be divisible by nbr_piles.
        :param random_: If random is True, the cards will be shuffled before splitting into piles.
        :return a list (of length 'nbr_piles') of lists of 'Card' instances.
        """
        if len(self) % nbr_piles != 0:
            raise ValueError("The decks size ({}) must be divisible by 'nbr_piles' ({}).".format(len(self), nbr_piles))
        pile_size = int(len(self) / nbr_piles)
        cards_to_distribute = sorted(self)
        if random_:
            rnd.shuffle(cards_to_distribute)
        pile_list = list()
        for k in range(nbr_piles):
            from_ = k*pile_size
            to = k*pile_size + pile_size
            pile_list.append(list(cards_to_distribute[from_: to]))
        return pile_list


# ######################### Combinations #########################

class Combination(metaclass=abc.ABCMeta):

    __slots__ = ('_cards', '_ranks')

    def __init__(self, cards: Iterable[Card]):
        try:
            next(iter(cards))
        except StopIteration:
            raise ValueError("Combination can not be empty!")

        check_all_isinstance(cards, Card)
        self._cards = CardSet(cards)
        check_true(len(self._cards) == len(list(cards)))
        self._ranks = {c.card_rank for c in self._cards}

    @property
    def cards(self)->CardSet:
        return self._cards

    @property
    @abc.abstractmethod
    def height(self)->int:
        raise NotImplementedError()

    @property
    def points(self)->int:
        return sum(c.points for c in self._cards)

    @staticmethod
    def make(cards):
        """
        makes a combination out of the given cards
        :param cards: the cards
        :return: the Combination
        :raise ValueError: if cards don't make a valid combination
        """
        nbr_cards = len(cards)
        err = None
        try:
            check_param(0 < nbr_cards <= 15, nbr_cards)
            if nbr_cards == 1:
                return Single(*cards)

            if nbr_cards == 2:
                return Pair(*cards)

            if nbr_cards == 3:
                return Trio(*cards)

            if nbr_cards % 2 == 0:
                with ignored(Exception):
                    ps = PairSteps.from_cards(cards)
                    return ps

            if nbr_cards == 4:
                return SquareBomb(*cards)

            if nbr_cards == 5:
                with ignored(Exception):
                    fh = FullHouse.from_cards(cards)
                    return fh

            if nbr_cards >= 5:
                st, sb = None, None
                with ignored(Exception):
                    st = Straight(cards)
                    sb = StraightBomb(st)
                if sb:
                    return sb
                if st:
                    return st

        except Exception as e:
            err = e
        raise ValueError("Is no combination: {}\ncards: {}".format(err, str(cards)))

    def can_be_played_on(self, other: Optional['Combination']) -> bool:
        if other is None:
            return True
        # handle bombs
        if isinstance(other, Bomb):
            pass
        # can only play on same type
        if not isinstance(other, type(self)):
            return False

        # return
        return self._can_be_played_on(other)

    @abc.abstractmethod
    def _can_be_played_on(self, other: 'Combination')->bool:
        """
        It is guaranteed that **isinstance(other, type(self))** is True.
        
        :param other: the other combination
        :return: True if this combination can be played on other
        """

    def contains_phoenix(self):
        return Card.PHOENIX in self._cards

    def contains_dragon(self):
        return Card.DRAGON in self._cards

    def issubset(self, other: CardSet)->bool:
        return self._cards.issubset(other)

    def fulfills_wish(self, wish: CardRank)->bool:
        return self.contains_cardrank(wish)

    def contains_cardrank(self, cardrank: CardRank)->bool:
        return cardrank in self._ranks

    def is_bomb(self)->bool:
        return False

    def __iter__(self):
        return iter(self._cards)

    def __eq__(self, other: 'Combination')->bool:
        return self.__class__ is other.__class__ and self.cards == other.cards and self.height == other.height

    def __hash__(self)->int:
        return hash(self._cards)

    def __len__(self)->int:
        return len(self._cards)

    def __contains__(self, card: Card)->bool:
        return self._cards.__contains__(card)

    def __str__(self):
        return "{}({})".format(self.__class__.__name__.upper(), str(self._cards))

    def __repr__(self)->str:
        return "{}({})".format(self.__class__, repr(self._cards))


class Single(Combination):

    __slots__ = ("_card", "_height")

    def __init__(self, card: Card):

        super().__init__([card])
        self._card = card
        self._height = self._card.card_height

    @property
    def card(self):
        return self._card

    @property
    def rank(self):
        return self._card.rank

    @property
    def height(self):
        return self._height

    def set_phoenix_height(self, newheight: Union[int, float]):
        """
        Set the height of tis single to the given height ONLY IF the Phoenix is the card of this single.
        Otherwise the call is ignored and a warning is printed.
        :param newheight:
        :return: the height of the single after this call
        """
        check_isinstance(newheight, (int, float))
        check_param(newheight == Card.PHOENIX.card_height or 2 <= newheight < 15, param=newheight)  # newheight must be between 2 and 14 (TWO and As)
        if self._card is Card.PHOENIX:
            self._height = newheight
        return self.height

    def is_phoenix(self):
        return self._card is Card.PHOENIX

    def contains_cardrank(self, cardrank: CardRank):
        return cardrank is self._card.rank

    def _can_be_played_on(self, other: 'Single')->bool:

        # other is guaranteed to be Single
        if self.card is Card.DRAGON:  # dragon can be played on all singles
            assert other.card is not Card.DRAGON
            return True
        if self.card is Card.DOG:  # dog can't be played on any single
            assert other.card is not Card.DOG
            return False

        return self.height > other.height  # must be bigger than other single

    def __contains__(self, card: Card):
        return self._card is card

# predefined Single Combinations
DOG_COMBINATION = Single(Card.DOG)
DRAGON_COMBINATION = Single(Card.DRAGON)


class Pair(Combination):

    __slots__ = ("_height", "_rank")

    def __init__(self, card1: Card, card2: Card):
        check_param(card1 is not card2, param=(card1, card2))  # different cards
        super().__init__((card1, card2))

        if Card.PHOENIX in self._cards:
            if card1 is Card.PHOENIX:
                card1, card2 = card2, card1  # make sure card1 is not Phoenix
            check_param(card1.suit is not CardSuit.SPECIAL, card1)
        else:
            check_param(card1.rank is card2.rank, (card1, card2))  # same value

        self._height = card1.card_height
        self._rank = card1.rank

    @property
    def height(self)->int:
        return self._height

    @property
    def rank(self):
        return self._rank

    def _can_be_played_on(self, other: 'Pair') -> bool:
        return self.height > other.height


class Trio(Combination):

    __slots__ = ("_height", "_rank")

    def __init__(self, card1: Card, card2: Card, card3: Card):
        check_param(card1 is not card2 and card1 is not card3 and card2 is not card3, param=(card1, card2, card3))  # 3 different cards
        super().__init__((card1, card2, card3))

        if Card.PHOENIX in self._cards:
            if card1 is Card.PHOENIX:
                card1, card2 = card2, card1  # make sure card1 is not Phoenix
            check_param(card1.suit is not CardSuit.SPECIAL)
        else:
            check_param(card1.rank is card2.rank is card3.rank)  # same ranks

        self._height = card1.card_height
        self._rank = card1.rank

    @property
    def height(self):
        return self._height

    @property
    def rank(self):
        return self._rank

    def _can_be_played_on(self, other: 'Trio') -> bool:
        return self.height > other.height


class FullHouse(Combination):

    __slots__ = ("_pair", "_trio", "_height")

    def __init__(self, pair: Pair, trio: Trio):
        check_isinstance(pair, Pair)
        check_isinstance(trio, Trio)
        check_param(not(pair.contains_phoenix() and trio.contains_phoenix()))  # phoenix can only be used once
        cards = set(itertools.chain(pair.cards, trio.cards))
        check_param(len(cards) == 5, param=(pair, trio))
        super().__init__(cards)
        self._height = trio.height
        self._pair = pair
        self._trio = trio

    @property
    def height(self)->int:
        return self._height

    @property
    def trio(self):
        return self._trio

    @property
    def pair(self):
        return self._pair

    @classmethod
    def from_cards(cls, cards: Iterable[Card]):
        check_param(len(set(cards)) == 5)  # 5 different cards
        check_param(Card.PHOENIX not in cards, "can't make from cards when Phoenix is present")
        pair = None
        trio = None
        for cs in CardSet(cards).rank_dict(values_only=True):
            if len(cs) == 2:
                pair = Pair(*cs)
            elif len(cs) == 3:
                trio = Trio(*cs)
            else:
                check_true(len(cs) == 0, ex=ValueError, msg="there is no fullhouse in the cards (cards: {})".format(cards))  # if this fails, then there is no fullhouse in the cards
        return cls(pair, trio)

    def _can_be_played_on(self, other: 'FullHouse') -> bool:
        return len(self) == len(other) and self.height > other.height

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.trio == other.trio and self.pair == other.pair

    def __hash__(self):
        return hash((self._trio, self._pair))

    def __str__(self):
        return "{}(<{}><{}>)".format(self.__class__.__name__.upper(), ",".join(map(str, sorted(self._trio))), ",".join(map(str, sorted(self._pair))))


class PairSteps(Combination):

    __slots__ = ("_lowest_pair_height", "_height", "_pairs", "_highest_rank")

    def __init__(self, pairs: Collection[Pair]):
        check_param(len(pairs) >= 2)
        check_all_isinstance(pairs, Pair)

        pairheights: Set[int] = {p.height for p in pairs}
        check_param(len(pairheights) == len(pairs))  # all pairs have different height
        check_param(max(pairheights) - min(pairheights) + 1 == len(pairs))  # pairs are consecutive

        cards = set(itertools.chain(*[p.cards for p in pairs]))
        check_param(len(cards) == 2*len(pairs), param=pairs)  # no duplicated card (takes care of multiple phoenix use)
        super().__init__(cards)
        self._height = max(pairheights)
        self._lowest_pair_height = min(pairheights)
        self._highest_rank = max(p.rank for p in pairs)
        self._pairs: Tuple[Pair, ...] = tuple(pairs)

    @property
    def height(self)->int:
        return self._height

    @property
    def pairs(self)->list:
        return list(self._pairs)

    @property
    def lowest_card_height(self)->int:
        return self._lowest_pair_height

    @property
    def highest(self)->CardRank:
        return self._highest_rank

    @classmethod
    def from_cards(cls, cards: Collection[Card]):
        check_param(len(cards) >= 4 and len(cards) % 2 == 0)
        check_param(Card.PHOENIX not in cards, "Can't make pairstep from cards when Phoenix is present")
        pairs = []
        for cs in CardSet(cards).rank_dict().values():
            if len(cs) == 2:
                pairs.append(Pair(*cs))
            check_true(len(cs) == 0, ex=ValueError, msg="Not a pairstep")
        return cls(pairs)

    def extend(self, pair: Pair):
        return PairSteps(self._pairs + (pair,))

    def _can_be_played_on(self, other: 'PairSteps') -> bool:
        return len(self) == len(other) and self.height > other.height

    def __str__(self):
        return "{}({})".format(self.__class__.__name__.upper(), ", ".join("{c[0]}{c[1]}".format(c=sorted(p.cards)) for p in self._pairs))

    def __len__(self):
        return len(self._pairs)


class Straight(Combination):

    __slots__ = ("_height", "_ph_as", '_lowest_rank', '_highest_rank')

    def __init__(self, cards: Iterable[Card], phoenix_as: Optional[Card]=None):
        # TODO speed!
        cards = set(cards)
        check_param(len(cards) >= 5)
        if Card.PHOENIX in cards:
            check_isinstance(phoenix_as, Card)
            check_param(phoenix_as not in cards)
            check_param(phoenix_as.suit is not CardSuit.SPECIAL, param=(phoenix_as, cards))
        else:
            phoenix_as = None

        cards_phoenix_replaced = [c for c in cards if c is not Card.PHOENIX] + [phoenix_as] if phoenix_as else cards
        check_param(len({c.rank for c in cards_phoenix_replaced}) == len(cards_phoenix_replaced))  # different card ranks

        self._lowest_rank = min(cards_phoenix_replaced).rank
        self._highest_rank = max(cards_phoenix_replaced).rank

        check_param(self._highest_rank.value - self._lowest_rank.value + 1 == len(cards_phoenix_replaced), param=cards)  # cards are consecutive

        super().__init__(cards)
        self._height = self._highest_rank.value
        self._ph_as = phoenix_as

    @property
    def height(self):
        return self._height

    @property
    def phoenix_as(self)->Card:
        return self._ph_as

    @property
    def lowest(self)->CardRank:
        return self._lowest_rank

    @property
    def highest(self) -> CardRank:
        return self._highest_rank

    def _can_be_played_on(self, other: 'Straight') -> bool:
        return len(self) == len(other) and self.height > other.height

    def __eq__(self, other):
        if self.contains_phoenix():
            return super().__eq__(other) and self.phoenix_as.card_rank is other.phoenix_as.card_rank
        else:
            return super().__eq__(other)

    def __str__(self):
        if self.contains_phoenix():
            return "{}({})".format(self.__class__.__name__.upper(), ",".join(str(c)+":"+str(self.phoenix_as) if c is Card.PHOENIX else str(c) for c in sorted(self._cards)))
        else:
            return super().__str__()

    def __hash__(self):
        if self.contains_phoenix():
            return hash((self._cards, self.height, self.phoenix_as.rank))
        else:
            return hash((self._cards, self.height))


class Bomb(Combination, metaclass=abc.ABCMeta):
    """
    Helps to make instance-checks for any bomb type
    """
    __slots__ = ()

    def is_bomb(self)->bool:
        return True


class SquareBomb(Bomb):

    __slots__ = ("_height", "_rank")

    def __init__(self, card1, card2, card3, card4):
        super().__init__((card1, card2, card3, card4))
        check_param(len(set(self.cards)) == 4)  # all cards are different
        # all cards have same card_value (takes also care of the phoenix)
        check_param(len({c.rank for c in self.cards}) == 1)
        self._height = card1.card_height + 500  # 500 to make sure it is higher than any other non bomb combination
        self._rank = card1.rank

    @property
    def height(self):
        return self._height

    @property
    def rank(self):
        return self._rank

    @classmethod
    def from_cards(cls, cards):
        return cls(*cards)

    def _can_be_played_on(self, other: 'SquareBomb') -> bool:
        return self.height > other.height


class StraightBomb(Bomb):

    __slots__ = ("_height", '_straight')

    def __init__(self, straight):
        check_isinstance(straight, Straight)
        thesuit = next(iter(straight)).suit
        check_true(all(c.suit is thesuit for c in straight))  # only one suit (takes also care of the phoenix)
        super().__init__(straight.cards)
        self._height = straight.height + 1000  # 1000 to make sure it is higher than any other non straightbomb
        self._straight = straight

    @property
    def height(self):
        return self._height

    @property
    def highest(self)->CardRank:
        return self._straight.highest

    @classmethod
    def from_cards(cls, *cards: Card):
        return cls(Straight(cards))

    def _can_be_played_on(self, other: 'StraightBomb') -> bool:
        return len(self) == len(other) and self.height > other.height

# ######################### General Combination #########################


class GeneralCombination(object):

    def __init__(self, type, height):
        check_param(len(height) == 2)
        self._type = type
        self._height = height

    @property
    def type(self):
        return self._type

    @property
    def height(self)->Tuple[int, Any]:
        return self._height

    @classmethod
    def from_combination(cls, combination: Combination):
        height = None
        if isinstance(combination, Single):
            height = GeneralCombination._single_height(combination)
        elif isinstance(combination, Pair):
            height = GeneralCombination._pair_height(combination)
        elif isinstance(combination, Trio):
            height = GeneralCombination._trio_height(combination)
        elif isinstance(combination, SquareBomb):
            height = GeneralCombination._squarebomb_height(combination)
        elif isinstance(combination, StraightBomb):
            height = GeneralCombination._straightbomb_height(combination)
        elif isinstance(combination, Straight):
            height = GeneralCombination._straight_height(combination)
        elif isinstance(combination, FullHouse):
            height = GeneralCombination._fullhouse_height(combination)
        elif isinstance(combination, PairSteps):
            height = GeneralCombination._pairsteps_height(combination)

        return GeneralCombination(type=combination.__class__, height=height)

    def find_in_cards(self, cards: Iterable[Card]):
        cards = CardSet(cards)
        try:
            myrank = CardRank.from_value(self.height[1])
            if self.type == Single:
                return next(cards.singles(contains_rank=myrank))
            if self.type == Pair:
                return next(cards.pairs(contains_rank=myrank))
            if self.type == Trio:
                return next(cards.trios(contains_rank=myrank))
            if self.type == SquareBomb:
                return next(cards.squarebombs(contains_rank=myrank))

            if self.type == StraightBomb:
                return next(st for st in cards.straightbombs() if GeneralCombination._straightbomb_height(st) == self.height)
            if self.type == Straight:
                return next(st for st in cards.straights() if GeneralCombination._straight_height(st) == self.height)
            if self.type == PairSteps:
                return next(ps for ps in cards.pairsteps() if GeneralCombination._pairsteps_height(ps) == self.height)
            if self.type == FullHouse:
                return next(fh for fh in cards.fullhouses() if GeneralCombination._fullhouse_height(fh) == self.height)

            raise LogicError("Should never reach here!")

        except StopIteration:
            raise ValueError("Can't find '{}' in '{}'".format(self, cards))

    @staticmethod
    def _single_height(single):
        return (1, single.rank.value)

    @staticmethod
    def _pair_height(pair):
        return (2, pair.rank.value)

    @staticmethod
    def _trio_height(trio):
        return (3, trio.rank.value)

    @staticmethod
    def _squarebomb_height(quad):
        return (4, quad.rank.value)

    @staticmethod
    def _fullhouse_height(fullhouse):
        return (5, fullhouse.trio.rank.value)

    @staticmethod
    def _straightbomb_height(straightbomb):
        return (len(straightbomb), straightbomb.highest.value)

    @staticmethod
    def _straight_height(straight):
        return (len(straight), straight.highest.value)

    @staticmethod
    def _pairsteps_height(pairstep):
        return (len(pairstep), pairstep.highest.value)

    def nbr_cards(self):
        if self._type == PairSteps:
            return self._height[0]*2
        else:
            return self._height[0]

    def __repr__(self):
        return "{me.__class__.__name__}(type: {me.type.__name__}, len: {me.height[0]}, height: {me.height[1]})".format(me=self)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((self._type, self._height))

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.type == other.type and self.height == other.height


def all_general_combinations_gen()->Generator[GeneralCombination, None, None]:
    special_ranks = {CardRank.DOG, CardRank.MAHJONG, CardRank.DRAGON, CardRank.PHOENIX}
    non_special_ranks = {r for r in CardRank if r not in special_ranks}
    # singles
    for rank in CardRank:
        yield GeneralCombination(type=Single, height=(1, rank.value))
    # pairs
    for rank in non_special_ranks:
        yield GeneralCombination(type=Pair, height=(2, rank.value))
    # trios
    for rank in non_special_ranks:
        yield GeneralCombination(type=Trio, height=(3, rank.value))
    # squarebombs
    for rank in non_special_ranks:
        yield GeneralCombination(type=SquareBomb, height=(4, rank.value))
    # fullhouse
    for rank_trio in non_special_ranks:
        yield GeneralCombination(type=FullHouse, height=(5, rank_trio.value))

    # straight
    for rank in non_special_ranks.union({CardRank.MAHJONG}):
        if rank >= CardRank.FIVE:  # streets can't end with a lower card than 5
            for l in range(5, rank.value + 1):  # iterate over all lengths possible for the straight ending in rank
                yield GeneralCombination(type=Straight, height=(l, rank.value))  # length, ending rank

    # pairsteps
    for rank in non_special_ranks:
        if rank >= CardRank.TWO:
            for l in range(2, rank.value):  # iterate over all lengths possible for the pairstep ending in rank
                yield GeneralCombination(type=PairSteps, height=(l, rank.value))  # length, ending rank

    # straightbombs (same as streets)
    for rank in non_special_ranks.union({CardRank.MAHJONG}):
        if rank >= CardRank.FIVE:  # streets can't end with a lower card than 5
            for l in range(5, rank.value + 1):  # iterate over all lengths possible for the straight ending in rank
                yield GeneralCombination(type=StraightBomb, height=(l, rank.value))  # length, ending rank
