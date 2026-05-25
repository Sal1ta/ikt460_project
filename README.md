# Chinese Checkers RL Project

Reinforcement learning agent for the IKT460 Chinese Checkers tournament.

The final player uses an afterstate value model with a small search wrapper.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run The GUI

```bash
python3 main.py gui
```

If the GUI does not appear, check for the Python window in the Dock or with
`Cmd + Tab`.

Demo video: [`media/gui-demo.mov`](media/gui-demo.mov)

## Run The Terminal Menu

```bash
python3 main.py
```

Then type `start`.

## Run The Tournament Player

```bash
python3 player.py
```

## Watch The Agent Locally

```bash
python3 main.py watch --players 4 --delay 0.01
```

## Check Submission

```bash
python3 scripts/checksubmission.py
```

## Main Files

- `player.py` final tournament client
- `main.py` local play and GUI
- `src/` game logic, agents, rewards, network, and GUI
- `scripts/` training and checking scripts
- `outputs/models/afterstate/` final model checkpoints
