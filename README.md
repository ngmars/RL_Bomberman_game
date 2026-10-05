# bomberman_rl

Reinforcement learning agents for the classic game Bomberman, built on the course framework for a student competition. The main result is a PPO agent with action masking (`agent_code/ppo_agent`) that wins 40% of four-player games and 64% of duels against the framework's `rule_based_agent`.

## Setup

The game runs on Python 3 with `numpy`, `pygame` and `tqdm`. The PPO agent additionally needs PyTorch and the packages pinned in `agent_code/ppo_agent/requirements.txt`:

```bash
pip install numpy pygame tqdm torch matplotlib pandas
```

```bash
pip install -r agent_code/ppo_agent/requirements.txt
```

A `Dockerfile` with the course's package set is included. It does not install `stable-baselines3`, `sb3-contrib` or `gymnasium`, so install the requirements file inside the container before running the PPO agent.

```bash
docker build -t bomberman .
```

Trained weights (`*.zip`) are excluded by `.gitignore`. To play the PPO agent, place a trained `model.zip` in `agent_code/ppo_agent/`, or train one as described below.

## Playing

Watch the PPO agent against three rule-based agents:

```bash
python main.py play --my-agent ppo_agent
```

Choose the line-up explicitly and run without the GUI:

```bash
python main.py play --agents ppo_agent rule_based_agent coin_collector_agent peaceful_agent --n-rounds 100 --no-gui --save-stats
```

Useful options for `play`:

| Option | Effect |
| --- | --- |
| `--scenario` | `classic` (tournament mode, default), `loot-crate`, `coin-heaven` or `empty` |
| `--n-rounds N` | Number of rounds, default 10 |
| `--train N` | Put the first N agents in training mode (for agents with a `train.py`) |
| `--no-gui` / `--skip-frames` | Run headless, or render only some steps |
| `--turn-based` | Wait for a key press before each step |
| `--save-replay` | Store the game for `python main.py replay <file>` |
| `--save-stats` | Write the results as JSON to `results/` |
| `--seed N` | Fix the world's random number generator |

To play yourself, add `user_agent` to `--agents`: arrow keys move, space drops a bomb, return waits.

## Agents

| Folder | Description |
| --- | --- |
| `ppo_agent` | Final agent: MaskablePPO with a CNN over the board and rule-based action masks |
| `ppo_agent_v1` | Earlier PPO version, kept as a baseline for the evaluation |
| `team_xys` | Submission copy of `ppo_agent` (inference files and `model.zip` only) |
| `dqn_agent` | DQN with replay buffer and target network |
| `my_agent` | Tabular Q-learning, the first attempt |
| `rule_based_agent`, `coin_collector_agent`, `peaceful_agent`, `random_agent` | Opponents provided by the framework |
| `tpl_agent`, `user_agent`, `fail_agent` | Framework template, keyboard-controlled agent and error-handling test agent |

## The PPO agent

**Observation.** `features.py` turns the game state into ten 17×17 planes: walls, crates, coins, own position, opponents, bombs, a danger map (how soon each tile is hit by a blast), active explosions, whether a bomb is available, and the opponents' positions in the previous step.

**Network.** `model.py` defines `BombermanCNN`, four 3×3 convolutions (32, 64, 64, 8 channels) followed by a 256-unit linear layer, used as the feature extractor for the policy and value heads.

**Action masking.** `legal_actions` in `features.py` restricts what the policy may choose. Each layer can be switched off with a constant at the top of the file:

| Switch | Effect |
| --- | --- |
| (always on) | No moves into walls, crates, bombs, other agents or tiles that are lethal in the next step |
| `SAFE_MASK` | Only actions after which the agent can still survive all known blasts, found by a search over the bomb timer horizon |
| `BLOCK_OTHERS` | Treat opponents as obstacles in that search |
| `BOMB_SLACK` | When an opponent is within `SLACK_RANGE` tiles, require an escape route that survives a one-step-earlier blast before bombing |
| `TRAP_RULE` | If a bomb would leave an opponent with no escape, force `BOMB` |
| `COIN_RULE` | If a coin is reachable and the agent is not in danger, allow only moves that shorten the path to it |

**Environment.** `env.py` wraps the game as a Gymnasium environment (`BombermanEnv`) with the learning agent in slot 0 and the chosen opponents in the others. The reward is the sum of configurable terms: game events (coins, kills, crates, death), a step cost, and shaping terms for idle waiting, backtracking, approaching the nearest goal and bombs that hit nothing.

## Training

```bash
python train_ppo_agent.py
```

The script runs eight parallel environments and is configured through the constants at its top: `SCENARIO`, `OPPONENT_SETS`, `REWARDS`, `TOTAL_STEPS`, `LOAD_FROM` (checkpoint to continue from) and `TB_NAME`. Checkpoints go to `agent_code/ppo_agent/checkpoints/`, and the final model is saved as `agent_code/ppo_agent/model.zip`. As committed, the script continues from an existing checkpoint; to train from scratch, use the commented-out `MaskablePPO(...)` constructor instead of `MaskablePPO.load(...)`.

Per-game metrics (score, coins, kills, suicides, bombs and others) are logged to TensorBoard:

```bash
tensorboard --logdir runs/ppo
```

The Q-learning and DQN agents train through the framework instead:

```bash
python main.py play --agents dqn_agent rule_based_agent --train 1 --no-gui --n-rounds 1000
```

`plot_training.py` and `plot_dqn.py` plot the metrics those two agents write to `training_metrics.csv`.

## Evaluation

```bash
python eval_ppo.py
```

This plays 300 rounds of the `classic` scenario per configuration and writes `results/report/rounds.csv` and `summary.csv`. A win means a strictly higher score than every opponent. Results against `rule_based_agent` (± is the 95% confidence interval):

| Configuration | Win rate, 4 players | Mean score, 4 players | Win rate, duel | Mean score, duel |
| --- | --- | --- | --- | --- |
| Full agent | 0.40 ± 0.06 | 4.59 | 0.64 ± 0.05 | 5.98 |
| Without `COIN_RULE` | 0.16 ± 0.04 | 2.57 | 0.20 ± 0.05 | 3.29 |
| Without `TRAP_RULE` | 0.31 ± 0.05 | 3.81 | 0.64 ± 0.05 | 5.71 |
| Without both rules | 0.13 ± 0.04 | 2.31 | 0.24 ± 0.05 | 3.45 |
| `ppo_agent_v1` | 0.17 ± 0.04 | 2.62 | 0.37 ± 0.05 | 4.07 |

Against three `coin_collector_agent` opponents the full agent wins 0.36 ± 0.05 of games with a mean score of 3.83.

Other analysis tools: `analyze_escape.py` compares sampled and greedy action selection when the agent is in danger, and `report_figure_gen.ipynb` produces the report figures.

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Command-line entry point for playing and replaying games |
| `environment.py`, `agents.py`, `items.py`, `events.py`, `settings.py` | Game framework: world logic, agent wrappers, bombs and coins, event names, game constants |
| `agent_code/` | One folder per agent, each with a `callbacks.py` providing `setup` and `act` |
| `train_ppo_agent.py`, `eval_ppo.py`, `analyze_escape.py` | PPO training, evaluation and analysis |
| `plot_training.py`, `plot_dqn.py` | Training curves for the Q-learning and DQN agents |
| `logs/`, `results/`, `runs/`, `replays/`, `screenshots/` | Generated output, not tracked |
