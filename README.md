# Retro-DQN
An implementation of a Deep Q Network used to train an AI to play Tetris.

## Requirements
The preferred version of Python is:

```
Python 3.8
```

The preferred installation of dependencies are from `pip`:

```shell
pip install -r requirements.txt
```

## Usage

You must install Python 3.8 and the dependencies in the requirements.txt file before trying to run the project. 

After both Python 3.8 and the dependencies have been installed, you can run a training session using:

```shell
python ./tetrisDQN_train.py
```

If you have an existing model and want to run a play session, you can use:

```shell
python ./tetrisDQN_play.py
```

An existing pre-trained model is included in the repository. To use the pre-trained model, rename tetris_best.pt to tetris.pt.
