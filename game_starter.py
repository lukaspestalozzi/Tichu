
import datetime
import gym

# this_folder = '/'.join(os.getcwd().split('/')[:])
# parent_folder = '/'.join(os.getcwd().split('/')[:-1])
#
# for p in [this_folder, parent_folder]:  # Adds the parent folder (ie. game) to the python path
#     if p not in sys.path:
#         sys.path.append(p)
from gamemanager import TichuGame
from gym_agents import *
from gym_agents.mcts import *
import logginginit

logger = logging.getLogger(__name__)


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


def random_against_random(target_points: int):
    game = create_agent_against_agent(RandomAgent, RandomAgent)
    res = game.start_game(target_points=target_points)
    return res


def balancedrandom_against_random(target_points: int):
    game = create_agent_against_agent(BalancedRandomAgent, RandomAgent)

    res = game.start_game(target_points=target_points)
    return res


def human_against_random(target_points: int):
    agents = [HumanInputAgent(position=0), RandomAgent(), RandomAgent(), RandomAgent()]
    game = TichuGame(*agents)

    res = game.start_game(target_points=target_points)
    return res


def human_against_ismcts(target_points: int):
    def make_agent():
        return BaseMonteCarloAgent(
                    make_best_ismctsearch(name='Best'),
                    iterations=100000, max_time=10, cheat=False
            )
    agents = [HumanInputAgent(position=0), make_agent(), make_agent(), make_agent()]
    game = TichuGame(*agents)

    res = game.start_game(target_points=target_points)
    return res


def makesearch_against_random(target_points: int):
    game = create_agent_against_agent(lambda: BaseMonteCarloAgent(make_ismctsearch(name='GameStarterMakesearchISMCTS',
                                                                                   nodeidpolicy=DefaultNodeIdPolicy,
                                                                                   determinizationpolicy=RandomDeterminePolicy,
                                                                                   treepolicy=UCBTreePolicy,
                                                                                   rolloutpolicy=RandomRolloutPolicy,
                                                                                   evaluationpolicy=RankingEvaluationPolicy,
                                                                                   bestactionpolicy=MostVisitedBestActionPolicy),
                                                                  iterations=100),
                                      BalancedRandomAgent)
    res = game.start_game(target_points=target_points)
    return res


def minimax_against_random(target_points: int):
    game = create_agent_against_agent(lambda: MinimaxAgent(depth=10),
                                      BalancedRandomAgent)
    res = game.start_game(target_points=target_points)
    return res


def minimax_against_mcts(target_points: int):
    game = create_agent_against_agent(lambda: MinimaxAgent(depth=10),
                                      lambda: BaseMonteCarloAgent(DefaultIsmcts(), iterations=1000, max_time=10))
    res = game.start_game(target_points=target_points)
    return res

if __name__ == "__main__":
    gym.undo_logger_setup()

    start_ftime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    this_folder = '/'.join(os.getcwd().split('/')[:])
    log_folder_name = "{}/logs/game_starter_{}".format(this_folder, start_ftime)

    logging_mode = logginginit.HumanplayMode
    logginginit.initialize_loggers(output_dir=log_folder_name, logging_mode=logging_mode, min_loglevel=logging.DEBUG)

    res = human_against_ismcts(target_points=1)
    print_game_outcome(res)
