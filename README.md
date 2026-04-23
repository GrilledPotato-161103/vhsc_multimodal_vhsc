<div align="center">

# Small project on learning model's uncertainty 
This is an unified codebase with Python to develop methodology for evaluate model's uncertainty and its cause from intermediate
<div align="left">

## File Structures: 
~~~
.
├── Makefile
├── README.md
├── configs (Contain components and experiments configurations)
├── data
│   └── checkpoints (Contain checkpoints for frozen model backbone)
├── logs (Experiment outputs and logs=(checkpoints, metrics, console, configs,... as packaged experiment))
├── notebooks
├── pyproject.toml
├── requirements.txt
├── scripts
│   └── schedule.sh
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── data (Code for data preparation through Lightning DataModule)
│   ├── models (Code for Model Processing and Inference Pipeline, including constructing code for backbones)
│   ├── plugins (Code for uncertainty modules and hook connectors)
│   |   ├── head (Code for nn.Module estimate endpoint uncertainty)
│   |   └── reconstructor (Code for nn.Module estimate missing modality and estimate input uncertainty)
│   ├── eval.py (Code for evaluation)
│   ├── train.py (Code for frozen backbone training)
│   ├── train_hook.py (Code for uncertainty and reconstructor training)
│   └── utils (Code for loggers and callback, containing customized Lightning Callbacks for uncertainty modules)
└── tests (Test workflow - (has not reach automation yet))
~~~
# How to use 
Every components in the codebase have their configurations managed by Hydra styled YAML, in which, contains 1. class name importable from pythonpath and env and 2. its properties. A component's property can also be a component, both will be recursively initialized by hydra.utils.instantiate(). 

Instantiation is done in src/train*.py and src/eval*.py python scripts, through a Hydra Config Manager Singleton called on context manager @hydra.main on wrapped function where the whole train/eval/test loops are done, in this case, by Lightning Framework. The 3 main Components and their roles are:
- Lightning DataModule: End2end data pre-processing factory from reading files to generate train/val/test Dataloader
- Lightning Module: End2End model loop manager from initialization, model inference to training, validation, test loop specification, weight update configuration through Optimizer and Scheduler
- Lightning Trainer: Immediate module connecting Main phase to external Utilities (Loggers, Callbacks, Helpers) throughout main phases (on_epoch_start, on_step, on_step_end,...)

## How to run
Expriments are implemented favoring WanDB Lightning Logger. You will have to login Wandb before running experiments, or turn off logging.
To train toy model, play 
~~~
python src/train.py <config overload>
~~~
In which, a toy model will be an neural estimator for the output of determined 2-variable mathematical expression. (reference to data configuration configs/data)

To evaluate toy model, play
~~~
python src/train.py <config overload>
~~~
To train hook modules, play
~~~
python src/train_hook.py <config overload>
~~~
Configuration overload can be done in runtime or in experiment configuration file referenced in @hydra.main, following Hydra convention for config overload.

