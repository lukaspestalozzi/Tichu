import abc
import argparse
import os
import sys
import time
import datetime

import gym
import logging


this_folder = '/'.join(os.getcwd().split('/')[:])
parent_folder = '/'.join(os.getcwd().split('/')[:-1])

for p in [this_folder, parent_folder]:  # Adds the parent folder (ie. game) to the python path
    if p not in sys.path:
        sys.path.append(p)


import logginginit

from gym_agents import (BalancedRandomAgent, DQNAgent2L_56x5, BaseMonteCarloAgent, DQNAgent4L_56x5,
                        DQNAgent2L_56x5_2_sep, DQNAgent2L_17x5_2_sep, DQNAgent2L_17x5_2)

from gym_tichu.envs.internals.utils import time_since
from gym_agents.keras_rl_utils import TichuSinglePlayerTrainEnv
from gym_agents.mcts import DefaultIsmcts

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    gym.undo_logger_setup()
    poss_envs = {'random': lambda agent: (BalancedRandomAgent(), agent, BalancedRandomAgent()),
                 'learned': lambda agent: (DQNAgent2L_56x5(), agent, DQNAgent2L_56x5()),
                 'learning': lambda agent: (agent, agent, agent),
                 'ismcts': lambda agent: (BaseMonteCarloAgent(DefaultIsmcts(), iterations=10000, max_time=2, cheat=True),
                                          agent,
                                          BaseMonteCarloAgent(DefaultIsmcts(), iterations=10000, max_time=2, cheat=True))}

    poss_agents = {'dqn_2l56x5': DQNAgent2L_56x5,  # 'dqn_4l56x5': DQNAgent4L_56x5,
                   'dqn_2l56x5_2_sep': DQNAgent2L_56x5_2_sep,
                   'dqn_2l17x5_2_sep': DQNAgent2L_17x5_2_sep, 'dqn_2l17x5_2': DQNAgent2L_17x5_2}
                   # 'sarsa_2l56x5': SarsaAgent2L_56x5}

    parser = argparse.ArgumentParser(description='Train Agent', allow_abbrev=False)

    # Agent
    parser.add_argument('agent', metavar='agent', type=str, choices=[k for k in poss_agents.keys()],
                        help='The agent to be trained. Choices: {}'.format(list(poss_agents.keys())))

    # ENV
    parser.add_argument('env', metavar='environment_name', type=str, choices=[k for k in poss_envs.keys()],
                        help='The name environment to rain on. Choices: {}'.format(list(poss_envs.keys())))

    # Steps
    parser.add_argument('steps', metavar='steps', type=int,
                        help='The number of steps to train for. (integer > 0)')

    # debuging
    parser.add_argument('--debug', dest='debug', required=False, action='store_true',
                        help='Flag, if present uses the DebugMode for logging.')

    args = parser.parse_args()
    print("train agent args: {}".format(str(args)))

    start_t = time.time()
    start_ftime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    NBR_STEPS = args.steps
    AGENT = poss_agents[args.agent]()
    ENV = TichuSinglePlayerTrainEnv(processor=AGENT.processor)
    ENV.configure(other_agents=poss_envs[args.env](AGENT))

    description = '{agentinfo}_{envn}'.format(agentinfo=AGENT.__class__.__name__, envn=args.env)

    # Folders
    # parent_folder = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
    this_folder = os.path.dirname(os.path.realpath(__file__))
    train_base_folder = '{parent_folder}/logs/{descr}_{t}_steps_{nbr}'.format(parent_folder=this_folder, t=start_ftime, nbr=NBR_STEPS, descr=description)

    log_folder_name = "{base}/my_logs".format(base=train_base_folder)

    # Logging
    logging_mode = logginginit.DebugMode if args.debug else logginginit.TrainMode
    logginginit.initialize_loggers(output_dir=log_folder_name, logging_mode=logging_mode, min_loglevel=logging.DEBUG)

    # Training
    print("Training Agent ({}) for {} steps ...".format(AGENT.__class__.__name__, NBR_STEPS))

    AGENT.train(env=ENV, base_folder=train_base_folder, nbr_steps=NBR_STEPS)

    print("Training time: {}".format(time_since(start_t)))
