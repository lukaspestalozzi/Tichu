# Tichu
Implementation of the Tichu game and agents able to play it.

-------------------------

## Dependencies
**Python 3.6+**

And following packages (all should be installable with pip or anaconda).

- **profilehooks**: To profile/measure function execution time (https://pypi.python.org/pypi/profilehooks)
- **keras-rl**: Neural Network Reinforcement learning (https://github.com/matthiasplappert/keras-rl). 
- **keras**: https://keras.io/.
- either **Tensorflow** or **Theano** (used by keras)
- **h5py**: For the h5f files.
- **gym**: OpenAI-gym (https://github.com/openai/gym#pip-version)
- **networkx**: For the Game-graph
- **numpy**
- **argparse**: To parse command line inputs
- **requests**: For the [tichumania](http://log.tichumania.de) scraper
- **BeautifulSoup**: For the [tichumania](http://log.tichumania.de) scraper

Then do (to register the gym-environment):
```bash
cd gym-tichu
pip install -e .
```

## Play a game
Gamelogs are written to the folder _Tichu/logs_

To play a game against three agents:
```bash
python play.py
```

To play a game against three agents and see all cards.
```bash
python play.py --cheat
```

To watch a game amongst four agents:
```bash
python play.py --lazy
```

More games can be found in the [game_starter.py](./game_starter.py)

## Train a Deep-Q-learning agent
Training results are written to the folder _nn_training/logs_

Example (train against random agents for 10000 steps (10000 decisions taken by the agent)): 
```bash
python nn_training/train_dqn.py dqn_2l17x5_2_sep random 10000
```

Following command shows all options
```bash
python nn_training/train_dqn.py -h
```

To visualize the training afterwards:
```bash
python nn_training/visualize_logs.py nn_training/logs/**/*.json
```

To save the plots:
```bash
python nn_training/visualize_logs.py nn_training/logs/**/*.json --save
```


# Run Experiments / Tournaments
Experiment results are written to the folder _experiments/logs_.

Launch the _**experiments/run_experiments.py**_ script.

For example, to launch a Tournament between the 4 DQN-agents, each game lasts until one team reached 1000 points, do:
```bash
python experiments/run_experiments.py nn_tournament --target 1000
```

To list all options:
```bash
python experiments/run_experiments.py -h
```