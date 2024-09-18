# RL-transformer
In-context Reinforcement Learning with GPT transformer

To run experiments:
- make sure to satisfy environment reqs (from environment.yml)
- run_config.yaml : can specify parameters for a new run (model dimensions, whether to regularize/symmetrize, training settings such as batch size, number of training steps, learning rate and etc);
results will be saved locally (model and losses) and also recorded in wandb as training progresses

- runscript.sh : submits the job of run_config to the schedulling system (deleted for annonymization purpose)

Files:

- main.py : model training, major chunk of algorithm here
- ddp_main.py : wrapper to run distributed data parallel training (not used in the end)
- training_utils.py : loss calculation, training schedulling, and empirical evaluation (via experiment function)
- transformer.py : implementation of symmetrized and regularized transformers
- env_MAB.py: implementation of bandit class
- my_algorithms.py: baseline algorithm (Thompson sampling) and the optimal policy (Gittins index)
