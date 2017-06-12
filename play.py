"""
Starts a game against the computer
"""
import argparse
import datetime
import gym
import sys
import os

import logging
from profilehooks import timecall

from gym_agents import BaseMonteCarloAgent, HumanInputAgent
from gym_agents.mcts import make_best_ismctsearch

this_folder = '/'.join(os.getcwd().split('/')[:])
parent_folder = '/'.join(os.getcwd().split('/')[:-1])

for p in [this_folder, parent_folder]:  # Adds the parent folder (ie. game) to the python path
    if p not in sys.path:
        sys.path.append(p)

from gamemanager import TichuGame
import logginginit

logger = logging.getLogger(__name__)


def make_ismcts_agent():
    return BaseMonteCarloAgent(
            make_best_ismctsearch(name='ISMCTS'),
            iterations=100000, max_time=10, cheat=False
    )


def print_game_outcome(outcome):
    assert len(outcome) == 2
    print("Final Result: {}".format(outcome[0]))
    rounds = outcome[1]
    for round in rounds:
        # round is a History object
        print("====================  New Round  ===================")
        print(round)

    print("Final Result: {}".format(outcome[0]))


def create_agent_against_agent(type1, type2)->TichuGame:
    agents = [type1(), type2(),
              type1(), type2()]
    return TichuGame(*agents)


def human_against_ismcts(target_points: int):
    agents = [HumanInputAgent(position=0), make_ismcts_agent(), make_ismcts_agent(), make_ismcts_agent()]
    game = TichuGame(*agents)

    res = game.start_game(target_points=target_points)
    return res


def ismcts_against_ismcts(target_points: int):

    agents = [make_ismcts_agent(), make_ismcts_agent(), make_ismcts_agent(), make_ismcts_agent()]
    game = TichuGame(*agents)

    res = game.start_game(target_points=target_points)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play', allow_abbrev=False)

    parser.add_argument('--target', dest='target_points', type=int, required=False, default=1000,
                        help='The number of points to play for')

    parser.add_argument('--lazy', dest='lazy', required=False, action='store_true',
                        help='When this flag is present, a game between 4 ISMCTS agents is started.')

    parser.add_argument('--cheat', dest='cheat', required=False, action='store_true',
                        help='When this flag is present, then you can see the handcards of the other players.')

    args = parser.parse_args()

    gym.undo_logger_setup()

    start_ftime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_folder_name = "{}/logs/game_starter_{}".format(this_folder, start_ftime)

    logging_mode = logginginit.HumanplayCheatMode if args.cheat or args.lazy else logginginit.HumanplayMode

    logginginit.initialize_loggers(output_dir=log_folder_name, logging_mode=logging_mode, min_loglevel=logging.DEBUG)

    if args.lazy:
        res = ismcts_against_ismcts(target_points=args.target_points)
    else:
        res = human_against_ismcts(target_points=args.target_points)
    print_game_outcome(res)
