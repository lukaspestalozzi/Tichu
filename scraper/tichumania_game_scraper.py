import traceback
from collections import defaultdict, Counter
from typing import Any, Union, Optional, Dict, List, Tuple, Generator
import datetime
import logging

import json
import functools
import pickle
import csv
from operator import itemgetter

import requests
import grequests as greq
import sys
import glob

import time
from bs4 import BeautifulSoup
import os
import errno
from collections import namedtuple

from itertools import islice

from gym_tichu.envs.internals.cards import Card as C, CardSet, GeneralCombination, all_general_combinations_gen, Card

logger = logging.getLogger(__name__)


class Player(namedtuple('Player', ['rank', 'name', 'nbr_games', 'nbr_won_games', 'elo'])):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other)


class GameOverview(namedtuple('GameOverview', ['date', 'p0', 'p1', 'p2', 'p3', 'result', 'won_rounds', 'highcards', 'bombs', 'tichus', 'grand_tichus', 'game_number'])):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return super().__eq__(other)


class Game(namedtuple('Game', ['game_overview', 'p0', 'p1', 'p2', 'p3', 'result', 'rounds'])):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __hash__(self):
        return hash((self.game_overview, self.result))

    def __eq__(self, other):
        return super().__eq__(other)


class Round(namedtuple('Round', ['initial_points', 'result', 'gt_hands', 'grand_tichus', 'tichus', 'trading_hands', 'traded_cards',
                                 'complete_hands', 'moves'])):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __hash__(self):
        return hash(tuple(self.moves))

    def __eq__(self, other):
        return super().__eq__(other)


class Move(namedtuple('Move', ['cards_before', 'player_name', 'cards_played', 'is_pass', 'is_clear', 'tichu', 'dragon_to'])):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __hash__(self):
        return hash((tuple(self.cards_before), self.player_name, tuple(self.cards_played), self.is_pass, self.is_clear, self.tichu, self.dragon_to))

    def __eq__(self, other):
        return super().__eq__(other)


class PlayedCards(object):
    """
    Stores the initial cards and the individual plays the player makes
    """

    def __init__(self):
        # self.playername = None
        self._initial_cards = set()
        self.plays = list()

    @property
    def initial_cards(self):
        return self._initial_cards

    @initial_cards.setter
    def initial_cards(self, cards):
        assert len(cards) == 14
        self._initial_cards = set(cards)

    def add_play(self, cards):
        # print("ADD PLAY: init: ", self.initial_cards, "add: ", cards)
        assert set(cards).issubset(self.initial_cards)
        # for c in cards:
        #     assert c not in flatten(self.plays)

        self.plays.append(cards)

    def to_dict(self):
        d = {'initial_cards': list(self.initial_cards), 'plays': list(self.plays)}
        assert len(self.plays) == len(d['plays'])
        # print(d)
        return d

    @classmethod
    def from_dict(cls, data):
        pc = PlayedCards()
        # pc.playername = data['playername']
        pc.initial_cards = set(data['initial_cards'])
        pc.plays = list(data['plays'])
        return pc

    def to_real_cards(self):
        new_pc = PlayedCards()
        new_pc.initial_cards = cards_class_list_to_cardsset(self.initial_cards)
        new_pc.plays = list(map(cards_class_list_to_cardsset, self.plays))
        return new_pc

    def iter_plays(self):
        """
        
        :return: generator yielding tuples (played cards, remaining handcards) for each play the player made
        """
        remainingcards = set(self.initial_cards)
        playedcards = list()
        yield playedcards, remainingcards
        for play in self.plays:
            playedcards.extend(play)
            for c in play:
                remainingcards.remove(c)
            yield playedcards, remainingcards

    def __repr__(self):
        return '{me.__class__.__name__}(init: {init}, plays: {pl})'.format(me=self, init=sorted(map(str, self.initial_cards)), pl='->'.join(map(str, self.plays)))



class_to_card_dict = {
        'c_00': C.DOG,
        'c_10': C.MAHJONG,
        'c_21': C.TWO_SWORD,
        'c_32': C.THREE_PAGODA,
        'c_43': C.FOUR_JADE,
        'c_54': C.FIVE_HOUSE,
        'c_61': C.SIX_SWORD,
        'c_72': C.SEVEN_PAGODA,
        'c_83': C.EIGHT_JADE,
        'c_94': C.NINE_HOUSE,
        'c_101': C.TEN_SWORD,
        'c_112': C.J_PAGODA,
        'c_123': C.Q_JADE,
        'c_134': C.K_HOUSE,
        'c_141': C.A_SWORD,
        'c_22': C.TWO_PAGODA,
        'c_33': C.THREE_JADE,
        'c_44': C.FOUR_HOUSE,
        'c_51': C.FIVE_SWORD,
        'c_62': C.SIX_PAGODA,
        'c_73': C.SEVEN_JADE,
        'c_84': C.EIGHT_HOUSE,
        'c_91': C.NINE_SWORD,
        'c_102': C.TEN_PAGODA,
        'c_113': C.J_JADE,
        'c_124': C.Q_HOUSE,
        'c_131': C.K_SWORD,
        'c_142': C.A_PAGODA,
        'c_23': C.TWO_JADE,
        'c_34': C.THREE_HOUSE,
        'c_41': C.FOUR_SWORD,
        'c_52': C.FIVE_PAGODA,
        'c_63': C.SIX_JADE,
        'c_74': C.SEVEN_HOUSE,
        'c_81': C.EIGHT_SWORD,
        'c_92': C.NINE_PAGODA,
        'c_103': C.TEN_JADE,
        'c_114': C.J_HOUSE,
        'c_121': C.Q_SWORD,
        'c_132': C.K_PAGODA,
        'c_143': C.A_JADE,
        'c_24': C.TWO_HOUSE,
        'c_31': C.THREE_SWORD,
        'c_42': C.FOUR_PAGODA,
        'c_53': C.FIVE_JADE,
        'c_64': C.SIX_HOUSE,
        'c_71': C.SEVEN_SWORD,
        'c_82': C.EIGHT_PAGODA,
        'c_93': C.NINE_JADE,
        'c_104': C.TEN_HOUSE,
        'c_111': C.J_SWORD,
        'c_122': C.Q_PAGODA,
        'c_133': C.K_JADE,
        'c_144': C.A_HOUSE,
        'c_150': C.PHOENIX,
        'c_160': C.DRAGON,
    }


def card_class_to_tichu_card(cclass: str)->Card:
    return class_to_card_dict[cclass]


def cards_class_list_to_cardsset(clist)->CardSet:
    return CardSet(map(card_class_to_tichu_card, clist))


def pretty_print_game(game: Game):
    print(game.game_overview)
    print()
    print(game.result)
    print(game.p0, 'and', game.p2, "vs.", game.p1, 'and', game.p3)
    print('rounds:')
    for round_ in game.rounds:
        print("=================================================================")
        print(round_.result)
        print('gt_hands', round_.gt_hands)
        print('grand_tichus', round_.grand_tichus)
        print('trading_hands', round_.trading_hands)
        print('traded_cards', round_.traded_cards)
        print('complete_hands', round_.complete_hands)
        print("moves:")
        for move in round_.moves:
            print("---------------------------------")
            print('player', move.player_name)
            print('cards_before', move.cards_before)
            print('cards_played', move.cards_played)
            print('is_pass:', move.is_pass, ', is_clear:', move.is_clear,
                  ', tichu:', move.tichu, ', dragon_to:', move.dragon_to)


def exceptions_to_warning(function):
    """
    A decorator that wraps the passed in function and warns the exception text, should one occur
    :return In case of an exception, the decorator/function returns None, otherwise whatever the function returns
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as ex:
            print(f"There was an exception in '{function.__name__}': text: ", repr(ex), "traceback: ", file=sys.stderr,
                  flush=True)
            traceback.print_tb(ex.__traceback__)

            return None

    return wrapper


def grouper(n, iterable):
    """

    :param n: integer > 0
    :param iterable: any iterable
    :return: Generator yielding tuples of size n from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            return
        yield chunk


def now():
    """
    :return: datetime.datetime.now()
    """
    return datetime.datetime.now()


def make_sure_path_exists(path):
    """
    Creates the folder if it does not exists yet.


    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class TichumaniaScraper(object):
    def __init__(self, base_url: str = 'http://log.tichumania.de/view/pages/', elolist_page: str = 'Elolist.php', games_page: str = 'Games.php', game_page: str = 'Game.php',
                 folder_path: str = '{}/tichumania_scraper_out/'.format(os.path.dirname(os.path.realpath(__file__))), scraped_gamenumbers_file: str = '.gamenumbers.json', scraped_gameoverviews_file: str = '.gameoverviews.json'):

        self.base_url = base_url
        self.elolist_page = elolist_page
        self.games_page = games_page
        self.game_page = game_page

        make_sure_path_exists(folder_path)  # creates the folder if it does not exists yet

        self.scraper_out_folder = folder_path
        self.scraped_gamenumbers_file = scraped_gamenumbers_file
        self.scraped_gameoverviews_file = scraped_gameoverviews_file
        self._out_folder_f = folder_path + '{}'

        self.scraped_gamenumbers = self._load_scraped_gamenumbers()
        self.scraped_gameoverviews = self._load_scraped_game_overviews()
        self._newly_scraped_games = set()

        print("nbr scraped_gamenumbers: ", len(self.scraped_gamenumbers))
        print("nbr scraped_gameoverviews: ", len(self.scraped_gameoverviews))

    @property
    def elolist_url(self):
        return f'{self.base_url}{self.elolist_page}'

    # ==================== Public Scrape Methods ====================

    def game_url(self, game_nbr):
        return '{self.base_url}{self.game_page}?game={game_nbr}'.format(self=self, game_nbr=game_nbr)

    def save_scraped_games_to_file(self, filename='scraper_out.json'):
        return self._write_to_file(self._newly_scraped_games, filename=filename)

    def scrape_games(self, max_nbr_games: int = None, max_time: int = None):
        """
        Scrapes games until either interrupted (by CTRL-C) or max_nbr_games is reached or max_time (in minutes) is reached.
        Skips games already scraped (games in the 'scraped_games' attribute)
        Stores the games in the folder determined by the 'scraper_out_folder/games/' attribute in the file '{current_time}_scraped_games.json'.

        :param max_nbr_games:
        :param max_time:
        :return: The number of scraped games
        """
        scraped_game_nbrs = set()
        nbr_scraped = 0
        try:
            gos_batch_gen = grouper(n=8, iterable=self.game_overviews_gen())
            if not max_nbr_games:
                max_nbr_games = float('inf')
            if not max_time:
                max_time = float('inf')

            start_t = time.time()
            end_t = start_t + max_time * 60
            while nbr_scraped < max_nbr_games:
                gos_batch = next(gos_batch_gen)
                games_batch = list(self._scrape_batch_of_games(gos_batch))
                # update various lists and sets
                game_nbrs = self._update_newly_scraped_games(games_batch)
                scraped_game_nbrs.update(game_nbrs)

                nbr_scraped += len(games_batch)
                if end_t < time.time():
                    break
            return nbr_scraped
        finally:
            # store scraped games
            make_sure_path_exists(self._out_folder_f.format('games/'))
            self._write_to_file([g for g in self._newly_scraped_games if g.game_overview.game_number in scraped_game_nbrs], filename=f'games/{now().strftime("%Y-%m-%d_%H:%M")}_scraped_games.json')
            # store scraped gamenumbers
            self._store_scraped_gamenumbers()

    def scrape_players(self, out_file: str = 'players.json'):
        """
        Scrapes all players and stores them to the file (in the folder determined by the 'scraper_out_folder' attribute.)

        :param out_file:
        :return: a list of all players.
        """
        players = list(self.find_best_players())
        self._write_to_file(players, filename=out_file)
        return players

    def scrape_gameoverviews(self, out_file: str = 'game_overviews.json', max_nbr_overviews: int = None, max_time: int = 1.0):
        """
        Scrapes gameoverviews (most recent first) until either interrupted (by CTRL-C) or max_nbr_games is reached or max_time (in minutes) is reached.
        And stores them to the out_file (in the folder determined by the 'scraper_out_folder' attribute.)

        :param out_file:
        :param max_nbr_overviews:
        :param max_time: [default None (unlimited time)] Time in Minutes after which the scraping should end (approximately).
        :return: The scraped game overviews
        """
        scraped_gos = []
        try:
            gos_gen = islice(self.game_overviews_gen(), max_nbr_overviews)
            if max_time:
                start_t = time.time()
                end_t = start_t + max_time * 60
                for go in gos_gen:
                    if go is not None:
                        scraped_gos.append(go)
                    if end_t < time.time():
                        break
            else:
                for go in gos_gen:
                    if go is not None:
                        scraped_gos.append(go)

        finally:
            self._write_to_file(scraped_gos, filename=out_file)

    def scrape_normal_handcards(self, max_time: float = 1.0):
        def store_handcards(hcrds):
            make_sure_path_exists(self._out_folder_f.format('handcards/'))
            self._write_to_file(list(hcrds), filename=f'handcards/{now().strftime("%Y-%m-%d_%H:%M:%S")}_scraped_handcards.json')

        def store_gos(gosses):
            make_sure_path_exists(self._out_folder_f.format('handcards/'))
            self._write_to_file(list(gosses), filename=f'handcards/{now().strftime("%Y-%m-%d_%H:%M:%S")}_scraped_gameoverviews.json')

        handcards = set()
        already_scraped_gos = set(gos_nbr for gos_nbr in self.all_stored_gos_nbrs_gen())
        print("already_scraped_gos length: {}".format(len(already_scraped_gos)))
        start_t = time.time()
        end_t = start_t + max_time * 60
        logger.info("scraping handcards for {} minutes... ".format(max_time))
        gos_batch_gen = grouper(n=8, iterable=self.game_overviews_gen())
        while time.time() < end_t:
            gos_batch = [gos for gos in next(gos_batch_gen)if gos.game_number not in already_scraped_gos]
            for handcard_tuple in self._scrape_normal_handcards_from_batch(gos_batch=gos_batch):
                handcards.add(handcard_tuple)
            already_scraped_gos.update(gos.game_number for gos in gos_batch)
            print("scraped {} handcards, time remaining: {:.2f}seconds".format(len(handcards), end_t - time.time()), flush=True)
            if len(handcards) > 10000:
                print("Store handcards & gosses in between.", flush=True)
                store_handcards(handcards)
                store_gos(already_scraped_gos)
                handcards = set()
        # end while

        if len(handcards) > 0:
            print("Store handcards & gosses at the end.", flush=True)
            store_handcards(handcards)
            store_gos(already_scraped_gos)

    def scrape_normal_handcards_batch_gen(self)->Generator[CardSet, None, None]:
        """
        :return: Generator yielding the handcards from 8 games at a time
        """
        gos_batch_gen = grouper(n=8, iterable=self.game_overviews_gen())
        while 1 != 0:  # while True
            gos_batch = next(gos_batch_gen)
            for hc in self._scrape_normal_handcards_from_batch(gos_batch=gos_batch):
                yield cards_class_list_to_cardsset(hc)

    def scrape_playedcards(self, max_nbr_scrape: int=1_000_000, max_time: float = 1.0):
        start_t = time.time()
        end_t = start_t + max_time * 60
        scraped_played_cards = set()
        nbr_saved = 0
        try:
            gos_batch_gen = grouper(n=8, iterable=self.game_overviews_gen())
            while time.time() < end_t and nbr_saved + len(scraped_played_cards) < max_nbr_scrape:
                gos_batch = [gos for gos in next(gos_batch_gen)]
                urls = [self.game_url(go.game_number) for go in gos_batch]
                if len(urls) == 0:
                    return  # nothing to do here

                rqsts = (greq.get(url) for url in urls)
                responses = greq.map(rqsts)
                failed_resp = [r for r in responses if not r.ok]
                if len(failed_resp) > 0:
                    print("Following requests failed: ", failed_resp)
                for gamesoup, go in ((BeautifulSoup(r.text, 'lxml'), go) for r, go in zip(responses, gos_batch) if r is not None):
                    # find player names in the game
                    names = [n.lower() for n in (go.p0, go.p1, go.p2, go.p3)]
                    # for each round
                    for round_tag in gamesoup.find_all('div', {'class': 'round'}):
                        played_cards_dict = {n: PlayedCards() for n in names}
                        # print("Round: ", BeautifulSoup.prettify(round_tag))
                        # find initial cards for each player
                        complete_hands = {k.lower(): v for k, v in self._scrape_complete_hands(round_tag).items()}
                        # print("Complete hands: ", complete_hands)
                        assert len(complete_hands) == 4  # 4 players
                        # same players as found before
                        assert all(k in played_cards_dict for k in complete_hands.keys()), 'comhands: {}, playedcards: {}'.format(complete_hands.keys(), played_cards_dict.keys())
                        for n, ch in complete_hands.items():
                            played_cards_dict[n].initial_cards = ch
                        # for each move in round:
                        for move_tag in round_tag.find_all('div', {'class': 'gameMove'}):
                            # print("Move: ", BeautifulSoup.prettify(move_tag))
                            # find player that played
                            player_name = move_tag.find('span', {'class': 'name'}).find('span').text
                            # find cards that he played
                            cards_tag = move_tag.find('div', {'class': 'cards'})
                            if cards_tag is not None:
                                # print("Cards Tag: ", BeautifulSoup.prettify(cards_tag))
                                # cards_before_move = [c_span['class'][-1] for c_span in cards_tag.find_all('span', {'class': 'card'})]
                                cards_played = [c_span['class'][-1] for c_span in cards_tag.find_all('span', {'class': 'played'})]
                                # print('Cards played: ', cards_played)
                                # add the cards to the played_cards
                                if len(cards_played):
                                    played_cards_dict[player_name.lower()].add_play(cards_played)
                        # save played cards
                        scraped_played_cards.update(played_cards_dict.values())
                    # endfor

                    # periodically save to file
                    if len(scraped_played_cards) > 10000:
                        nbr_saved += len(scraped_played_cards)
                        self._write_to_file(data=[pc.to_dict() for pc in scraped_played_cards], filename='played_cards{}.json'.format(time.time()))
                        scraped_played_cards = set()

                #endfor
            # end while
        finally:
            # store the remaining scraped
            self._write_to_file(data=[pc.to_dict() for pc in scraped_played_cards], filename='played_cards{}.json'.format(time.time()))


    # ==================== Public File Handling Methods ====================

    def all_stored_games_gen(self):
        """

        :return: Generator yielding each game that has been scraped (and is in the scraper_out_folder/games/ folder).
        """
        game_files = glob.glob(self._out_folder_f.format('games/*scraped_games*'))
        for filename in game_files:
            games = self._load_from_file(filename, default=[])
            for game in games:
                yield game

    def all_stored_handcards_gen(self):
        handcard_files = glob.glob(self._out_folder_f.format('handcards/*handcards*'))
        for filename in handcard_files:
            yield from self._load_from_file(filename, default=[])

    def all_stored_gos_nbrs_gen(self):
        gos_files = glob.glob(self._out_folder_f.format('handcards/*gameoverviews*'))
        for filename in gos_files:
            yield from map(int, self._load_from_file(filename, default=[]))

    # ==================== Private Scrape Methods ====================

    def _write_to_file(self, data, filename: str):
        """
        Writes the data to the file

        :param data:
        :param filename:
        :return:
        """
        file_path = self._out_folder_f.format(filename)
        with open(file_path, 'w') as f:
            print("Writing to file ", file_path, "data of type ", type(data), " and of length: ", len(data) if data.__len__ else 'Has no length.')
            json.dump(data, f)

    def _load_from_file(self, filename: str, default=None):
        """
        Loads data from the given file. Must have been written with the '_write_to_file' function.

        :param filename: the filename
        :param default: The value returned if the file does not exist
        :return: the object in the file or none if the file does not exist
        """
        file_path = self._out_folder_f.format(filename) if self._out_folder_f.format("") not in filename else filename
        try:
            with open(file_path, 'r') as f:
                print("Loading file ", file_path, '... ', end='', flush=True)
                data = json.load(f)
                print("loaded data of type ", type(data), " and of length: ", len(data) if data.__len__ else 'Has no length.')
                return data
        except FileNotFoundError as fnfe:
            print("file '{}' did not exist, returning default: {}".format(file_path, default))
            return default

    def _store_scraped_gamenumbers(self):
        self._write_to_file(list(self.scraped_gamenumbers), self.scraped_gamenumbers_file)

    def _load_scraped_gamenumbers(self):
        """

        :return: A set of the numbers of the already scraped games.
        """
        gamenumbers = set(self._load_from_file(self.scraped_gamenumbers_file, default=set()))
        return gamenumbers

    def _load_scraped_game_overviews(self):
        """

        :return: a set of the already scraped gameoverviews.
        """
        gameoverviews = set(self._load_from_file(self.scraped_gameoverviews_file, default=set()))
        return gameoverviews

    def _update_newly_scraped_games(self, games):
        """
        Keeps the internal state of the newly scraped games and scraped game numbers consistent.

        :param games:
        :return: a set containing the game numbers of the given games
        """
        if not games.__iter__:
            games = [games]

        to_add = {g for g in games}
        self._newly_scraped_games.update(to_add)
        game_nbrs = {g.game_overview.game_number for g in games}
        self.scraped_gamenumbers.update(game_nbrs)
        return game_nbrs

    def _games_url(self, page_from=0, amount=100, player_name=None):
        gurl = '{self.base_url}games/content.php?start={page_from}&count={amount}'.format(self=self,
                                                                                          page_from=page_from,
                                                                                          amount=amount)
        if player_name:
            gurl += f'&player={player_name}'
        return gurl

    # ==================== Scraping Methods ====================

    # -------------------- players --------------------
    def find_best_players(self):
        """
        :return:Generator yielding players (ranked by elo score) Note: in April 2017 there were almost 5000 players
        """
        r = requests.get(url=self.elolist_url)
        soup = BeautifulSoup(r.text, 'lxml')

        for row in soup.find_all('tr'):
            player = self._player_from_soup(row)
            if player is not None:
                yield player

    @exceptions_to_warning
    def _player_from_soup(self, row_soup):
        col = row_soup.find_all('td')
        if len(col):
            t = [e.text for e in col]
            p = Player(rank=int(t[0]), name=t[1], nbr_games=int(t[2]), nbr_won_games=t[3], elo=int(t[4]))
            return p
        else:
            return None

    # -------------------- game overviews --------------------
    def game_overviews_gen(self, player_name: str = None):
        """

        :param player_name: a playername, if not None, then only returns gameoverviews where the player with this name appears in.
        :return Generator yielding GameOverviews (most recent first).
        """
        all_rows = ['dummy_element']
        next_from = 0
        amount = 100
        while len(all_rows):
            soup = BeautifulSoup(requests.get(url=self._games_url(next_from, amount, player_name=player_name)).text, 'lxml')
            next_from += amount
            all_rows = soup.find_all('tr')
            for row_soup in all_rows:
                go = self._game_overview_from_soup(row_soup)
                if go is not None:
                    yield go

    @exceptions_to_warning
    def _game_overview_from_soup(self, row_soup):
        # TODO check cache
        team0 = tuple([span.text.strip() for span in row_soup.find('td', {'title': 'Home team'}).find_all('span')])
        team1 = tuple([span.text.strip() for span in row_soup.find('td', {'title': 'Guest team'}).find_all('span')])
        go = GameOverview(date=row_soup.find('td', {'title': 'Date'}).text.strip(),
                          p0=team0[0], p1=team1[0],
                          p2=team0[1], p3=team1[1],
                          result=self._read_score(row_soup.find('td', {'title': 'Result'})),
                          won_rounds=self._read_score(row_soup.find('td', {'title': 'Won rounds'})),
                          highcards=self._read_score(row_soup.find('td', {'title': 'Highcards'})),
                          bombs=self._read_score(row_soup.find('td', {'title': 'Bombs'})),
                          tichus=self._read_score(row_soup.find('td', {'title': 'Tichus'})),
                          grand_tichus=self._read_score(row_soup.find('td', {'title': 'Grand Tichus'})),
                          game_number=row_soup.find('a', {'class': 'gameLink'}).get('href').split('=')[-1])
        assert all(e is not None for e in go)
        return go

    # -------------------- games --------------------
    """
    def games_for_playername(self, name, save_periodicaly=100):

        Generator yielding Games (most recent first) where the player with the given name played in.

        :param name: player name
        :param save_periodicaly: integer. calls 'save_scraped_games_to_file' after scraping the given amount of games. set negative to disable

        gos_gen = self.game_overviews_for_playername(name)
        for gos in grouper(8, gos_gen):
            for game in self._scrape_batch_of_games(gos):
                if game is not None:
                    if 0 < save_periodicaly and len(self.scraped_games) % save_periodicaly == 0:
                        self.save_scraped_games_to_file(self.periodical_saves_file)
                    yield game
    """

    def _scrape_batch_of_games(self, game_overviews):
        """
        Sends the requests for the games asynchronously and then parses the games from the responses.
        Already scraped games are skipped.

        Note: tichumania has a limit of around 8 simultaneous requests from the same user. so len(game_overviews) should not be bigger than 8

        :param game_overviews:
        :return: Generator yielding the individual games for the game_overviews
        """

        gos = list()
        skipped = 0  # counts the number of games skipped because already scraped before.
        for go in game_overviews:
            if go.game_number not in self.scraped_gamenumbers:
                gos.append(go)
            else:
                skipped += 1
        # if skipped > 0:
            # print(f'skipped {skipped} games')

        urls = [self.game_url(go.game_number) for go in gos]
        if len(urls) == 0:
            return  # nothing to do here

        rqsts = (greq.get(url) for url in urls)
        responses = greq.map(rqsts)
        failed_resp = [r for r in responses if not r.ok]
        if len(failed_resp) > 0:
            print("Following requests failed: ", failed_resp)
        for game in (self.scrape_game_from_soup(BeautifulSoup(r.text, 'lxml'), go) for r, go in zip(responses, gos) if r is not None):
            if game is not None:
                yield game

    @exceptions_to_warning
    def _scrape_normal_handcards_from_batch(self, gos_batch):
        urls = [self.game_url(go.game_number) for go in gos_batch]
        if len(urls) == 0:
            return  # nothing to do here

        rqsts = (greq.get(url) for url in urls)
        responses = greq.map(rqsts)
        failed_resp = [r for r in responses if not r.ok]
        if len(failed_resp) > 0:
            print("Following requests failed: ", failed_resp)
        for handcards_in_game_gen in (self.scrape_normal_hands_from_soup(BeautifulSoup(r.text, 'lxml'), go) for r, go in zip(responses, gos_batch) if r is not None):
            if handcards_in_game_gen is not None:
                yield from handcards_in_game_gen

    @exceptions_to_warning
    def scrape_game(self, game_overview):
        """
        Scrapes the game corresponding to the overview from the server.

        Note: Does NOT store the game anywhere.

        :param game_overview:
        :return: The Game corresponding to the given GameOverview.
        :returns None: if either some exception occured or the game already was scraped before.
        """

        if game_overview.game_number in self.scraped_gamenumbers:
            # print("already fetched ", game_overview.game_number, ':)')
            return None

        r = requests.get(url=self.game_url(game_overview.game_number))
        game_soup = BeautifulSoup(r.text, 'lxml')
        game = self.scrape_game_from_soup(game_soup, game_overview)

        return game

    @exceptions_to_warning
    def scrape_game_from_soup(self, game_soup, game_overview):
        """

        :param game_soup: The beautifulsoup object of the game page
        :param game_overview:
        :return: The Game corresponding to the given game_soup
        """
        print(f"parsing game {game_overview.game_number} ... ", end='')
        start_t = time.time()
        # print(game_soup.prettify())
        game_data = {
            'game_overview': game_overview,
            'p0': game_overview.p0,
            'p1': game_overview.p1,
            'p2': game_overview.p2,
            'p3': game_overview.p3,
            'result': game_soup.find('span', {'class': 'gameResult'}).text
        }

        rounds = [self._scrape_round(round_tag) for round_tag in game_soup.find_all('div', {'class': 'round'})]

        # create the Game
        game_data['rounds'] = rounds
        print('done. time:', time.time() - start_t)

        return Game(**game_data)

    @exceptions_to_warning
    def scrape_normal_hands_from_soup(self, game_soup, game_overview):
        all_handcards_tag = game_soup.find_all('div', {'class': 'cards'})
        hands = set()
        for cards_tag in all_handcards_tag:
            handcards_list = [c_span['class'][-1] for c_span in cards_tag.find_all('span', {'class': 'card'})]
            hands.add(tuple(sorted(handcards_list)))
        yield from hands

    def _scrape_round(self, round_soup):
        """

        :param round_soup: beautiful soup tag of the tichu round
        :return: The scraped Round described by the round_tag
        """
        round_data = {}

        # grand tichu hands
        gt_hands = self._scrape_grand_tichu_hands(round_soup)
        round_data['gt_hands'] = gt_hands

        # grand tichu
        round_data['grand_tichus'] = self._scrape_players_announced_grand_tichu(round_soup)

        # trading hands
        trading_hands = self._scrape_trading_hands(round_soup)
        round_data['trading_hands'] = {pl_name: t[0] for pl_name, t in trading_hands.items()}
        round_data['traded_cards'] = {pl_name: t[1:] for pl_name, t in trading_hands.items()}

        # complete hands
        complete_hands = self._scrape_complete_hands(round_soup)
        round_data['complete_hands'] = complete_hands

        # Moves
        moves = self._scrape_moves(round_soup)
        round_data['moves'] = moves
        round_data['tichus'] = tuple({m.player_name for m in moves if m.tichu})

        # round result
        initial_points_str = ''.join(rr.text for rr in round_soup.find_all('span', {'class': 'interimResult'}))
        result_str = ''.join(rr.text for rr in round_soup.find_all('span', {'class': 'roundResult'}))
        round_data['initial_points'] = tuple([int(n) for n in initial_points_str.strip().split(':')])
        round_data['result'] = tuple([int(n) for n in result_str.strip().split(':')])

        # Create Round
        return Round(**round_data)

    @staticmethod
    def _read_score(soup):
        # TODO make nicer
        return ':'.join(e.text.strip() for e in soup.find_all('span')[:2])

    @staticmethod
    def _scrape_grand_tichu_hands(round_soup):
        """

        :param round_soup:
        :return: a dictionary player_name -> tuple of cards
        """
        hands = {}
        gt_hands = round_soup.find('div', {'class': 'gtHands'})
        for hand in gt_hands.find_all('div', {'class': 'line'}):
            cards_tag = hand.find('div', {'class': 'cards'})
            player_name = hand.find('span', {'class': 'name'}).find('span').text  # TODO ev text from first span

            # cards
            cards = [c_span['class'][-1] for c_span in cards_tag.find_all('span')]
            hands[player_name] = cards

        return hands

    @staticmethod
    def _scrape_players_announced_grand_tichu(round_soup):
        """
        :param round_soup:
        :return: tuple containing the playernames of the players that announced grand tichu
        """
        gtichus = set()
        for gt_span in round_soup.find_all('span', {'class': 'gt'}):
            player_name = gt_span.parent.find('span', {'class': 'name'}).find(
                'span').text  # TODO ev text from first span
            gtichus.add(player_name)
        return tuple(gtichus)

    @staticmethod
    def _scrape_trading_hands(round_soup):
        """

        :param round_soup:
        :return: a dictionary player_name -> tuple(list of cards, traded card right, traded card teammate, traded card left)
        """
        trading_data = defaultdict(lambda: [None, None, None, None])
        trading_hands = round_soup.find('div', {'class': 'fullHands'})
        for hand in trading_hands.find_all('div', {'class': 'line'}):
            cards_tag = hand.find('div', {'class': 'cards'})
            player_name = hand.find('span', {'class': 'name'}).find('span').text  # TODO ev text from first span
            cards = [c_span['class'][-1] for c_span in cards_tag.find_all('span', {'class': 'card'})]
            trading_data[player_name][0] = tuple(cards)  # index 0 contains the handcards

            # traded cards
            traded_tags = hand.find_all('div', {'class': 'trading'})
            assert len(traded_tags) == 3, "traded_tags: " + traded_tags.prettify()
            for traded_t in traded_tags:
                trade_icon_tag = traded_t.find('span', {'class': 'tradeIcon'})
                icon_class = trade_icon_tag['class'][-1]
                player_offset = int(icon_class[-1])  # <span class="tradeIcon ti02"></span>
                traded_card = traded_t.find('span', {'class': 'card'})['class'][-1]
                trading_data[player_name][player_offset] = traded_card

                # print("icon_class", icon_class, "-> player_offset", player_offset, '=>', round_data['traded_cards'][player_name])

            # the list immutable
            trading_data[player_name] = tuple(trading_data[player_name])

        return trading_data

    @staticmethod
    def _scrape_complete_hands(round_soup)-> Dict[str, Tuple[str]]:
        """

        :param round_soup:
        :return: a dictionary player_name -> tuple of cards
        """
        hands = dict()
        complete_hands = round_soup.find('div', {'class': 'completeHands'})
        for hand in complete_hands.find_all('div', {'class': 'line'}):
            cards_tag = hand.find('div', {'class': 'cards'})
            player_name = hand.find('span', {'class': 'name'}).find('span').text  # TODO ev text from first span
            cards = [c_span['class'][-1] for c_span in cards_tag.find_all('span', {'class': 'card'})]
            hands[player_name] = tuple(cards)

        return hands

    @staticmethod
    def _scrape_moves(round_soup):
        """

        :param round_soup:
        :return: tuple of moves
        """
        moves = list()
        for move_tag in round_soup.find_all('div', {'class': 'gameMove'}):
            move_data = {'cards_before': None, 'cards_played': None}
            player_name = move_tag.find('span', {'class': 'name'}).find('span').text  # TODO ev text from first span
            move_data['player_name'] = player_name

            move_data['tichu'] = len(move_tag.find_all('span', {'class': 'tichu'})) > 0

            cards_tag = move_tag.find('div', {'class': 'cards'})
            if cards_tag is not None:
                cards_before_move = [c_span['class'][-1] for c_span in
                                     cards_tag.find_all('span', {'class': 'card'})]
                move_data['cards_before'] = cards_before_move

                cards_played = [c_span['class'][-1] for c_span in cards_tag.find_all('span', {'class': 'played'})]
                move_data['cards_played'] = cards_played

            subline_tag = move_tag.find('div', {'class': 'subline'})
            move_data['is_pass'] = 'Pass' in subline_tag.text
            move_data['is_clear'] = 'final' in subline_tag['class']
            move_data['dragon_to'] = subline_tag.text.strip().split(' ')[-1] if move_data['is_clear'] and 'Dragon' in subline_tag.text else None
            moves.append(Move(**move_data))

        return tuple(moves)


class GenCombWeights(object):

    def __init__(self, load: bool=True):
        self.filename = "./gcombweights.pkl"
        self._gcomb_counter: Dict[Tuple[int, GeneralCombination], int] = defaultdict(int)  # (length, gencomb) -> how many time the gcomb was seen for the length
        self._total_counter: Dict[int, int] = defaultdict(int) # how many gcombs have been seen for the given length
        if load:
            try:
                self.load_from_file()
            except FileNotFoundError:
                print("Loading GenCombWeights: File not Found {}, continuing empy.".format(self.filename))

    @property
    def weights(self):
        return self._calc_weights_dict()

    @staticmethod
    def weights_from_file(filename: str=None)->Dict[Tuple[int, GeneralCombination], float]:
        gcw = GenCombWeights(load=False)
        gcw.load_from_file(filename=filename)
        return gcw.weights

    def _calc_weights_dict(self)->Dict[Tuple[int, GeneralCombination], float]:
        weight_dict = defaultdict(float)
        for len_gcomb, nbr in self._gcomb_counter.items():
            try:
                weight_dict[len_gcomb] = nbr / self._total_counter[len_gcomb[0]]
            except ZeroDivisionError:
                pass
        return weight_dict

    def add_gcombs_for_length(self, length, gcombs):
        self._total_counter[length] += len(gcombs)
        for gcomb in gcombs:
            tup = (length, gcomb)
            self._gcomb_counter[tup] += 1

    def proba_for_gcomb_and_length(self, length, gcomb):
        try:
            return self._gcomb_counter[(length, gcomb)] / self._total_counter[length]
        except ZeroDivisionError:
            return 0

    def load_from_file(self, filename: str=None):
        if filename is None:
            filename = self.filename
        with open(filename, 'rb') as file:
            obj = pickle.load(file)
            self._gcomb_counter, self._total_counter = obj

    def save_to_file(self, filename: str=None):
        if filename is None:
            filename = self.filename
        obj = [self._gcomb_counter, self._total_counter]
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)

    def save_to_csv(self, filename: str=None):
        if filename is None:
            filename = "./weights_{}.csv".format(sum(self._total_counter.values()))
        print("writing to csv file: {} ...".format(filename), end="", flush=True)
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            csvwriter.writerow(['length', 'type', 'comblength', 'height', 'probability'])
            for len_gcomb, proba in self.weights.items():
                length, gcomb = len_gcomb
                csvwriter.writerow([length, gcomb.type.__name__, gcomb.height[0], gcomb.height[1], proba])
        print("...done", flush=True)

    def scrape_weights(self, max_nbr=500000, save: bool=True):
        print("scraping {} handcards...".format(max_nbr))
        start_t = time.time()
        scraper = TichumaniaScraper()
        for n, cards in enumerate(islice(scraper.scrape_normal_handcards_batch_gen(), max_nbr)):
            self.add_gcombs_for_length(length=len(cards), gcombs=cards.all_general_combinations())
            if n % 1000 == 0:
                print(" ... {} ".format(n), end="", flush=True)
            if n % 10000 == 0:
                print()
        if save:
            self.save_to_file()
        print("... done scraping. it took {} seconds".format(time.time() - start_t))

if __name__ == '__main__':
    SCRAPE = True
    ANALYZE = False
    scraper = TichumaniaScraper()
    if SCRAPE:
        scraper.scrape_playedcards(max_nbr_scrape=1000000, max_time=360)

    if ANALYZE:
        playedcards: List[PlayedCards] = [PlayedCards.from_dict(d) for d in scraper._load_from_file('played_cards_len296.json')]
        print(playedcards[:5])
        realpcs = [pc.to_real_cards() for pc in playedcards]
        print(realpcs[:5])





    # weights: GenCombWeights = GenCombWeights(load=False)
    # weights.scrape_weights(max_nbr=1000000)
    # weights.save_to_csv()
    # w = weights.weights
    #
    # for l in range(1, 15):
    #     for gcomb in all_general_combinations_gen():
    #         tup = (l, gcomb)
    #         proba = w[tup]
    #         if proba > 0:
    #             print("{a}: {b} -> {prob}".format(a=l, b=gcomb, prob=proba))

    # n_games, output_file = int(sys.argv[1]), sys.argv[2]
    # run_scraper(n_games, output_file)
    # scraper = TichumaniaScraper()
    # # scraper.scrape_normal_handcards(max_time=3.0)
    # cardsets = [cards_class_list_to_cardsset(clist) for clist in scraper.all_stored_handcards_gen()]
    # print("nbr handcards: ", len(cardsets))
    #
    # # ### group by length
    # len_dict = defaultdict(list)
    # for cset in cardsets:
    #     len_dict[len(cset)].append(cset)
    #
    # # print how many for each length
    # print("len -> how many handcards:")
    # how_many_for_each_length = {k: len(l) for k, l in len_dict.items()}
    # print('\n'.join("{}: {}".format(k, v) for k, v in sorted(how_many_for_each_length.items(), key=itemgetter(1), reverse=True)))
    #
    # #  ### map to all possible combinations and their general combinations
    # len_gcombs_dict = defaultdict(list)
    # for cset in cardsets:
    #     diff_combs = cset.possible_combinations()
    #     gen_combs = set(map(GeneralCombination.from_combination, diff_combs))
    #     len_gcombs_dict[len(cset)].extend(gen_combs)
    #
    # #  print how many gcombs for each length
    # print("len -> how many gcombs:")
    # how_many_for_each_length = {k: len(l) for k, l in len_gcombs_dict.items()}
    # print('\n'.join("{}: {}".format(k, v) for k, v in sorted(how_many_for_each_length.items(), key=itemgetter(1), reverse=True)))
    #
    # #  ### count how many identical gcombs in for each length
    # len_gcombs_counter = dict()
    # for length, gcombs in len_gcombs_dict.items():
    #     len_gcombs_counter[length] = Counter(gcombs)
    #
    # #  print how many same gcombs for each length and gcomb
    # print("len -> for each different gcomb how many:")
    # for length, counter in len_gcombs_counter.items():
    #     print(length, "(total ", how_many_for_each_length[length], "): ")
    #     for line in str(counter).split('GeneralCombination'):
    #         print(line)
    #     print("===========================================")
    #
    # # dict for each gencomb the probas of each length
    # gencomb_len_proba_dict = {gcomb: {l: 0.0 for l in range(1, 15)} for gcomb in all_general_combinations_gen()}
    # for gcomb in all_general_combinations_gen():
    #     for l in range(1, 15):
    #         total_gcombs = how_many_for_each_length[l]
    #         nbr_of_the_comb = len_gcombs_counter[l].get(gcomb, 0)
    #         proba = nbr_of_the_comb / total_gcombs
    #         gencomb_len_proba_dict[gcomb][l] = proba
    #
    # # print gencomb_len_proba_dict
    # for gcomb, l_dict in gencomb_len_proba_dict.items():
    #     print(gcomb, ":")
    #     for l, proba in l_dict.items():
    #         print("{indent} {l}: {proba}".format(indent="     ", l=l, proba=proba))



