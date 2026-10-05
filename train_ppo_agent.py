from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agent_code.ppo_agent import env
from agent_code.ppo_agent.env import BombermanEnv
from agent_code.ppo_agent.model import BombermanCNN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch

import events as e

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)

N_ENVS = 8
TOTAL_STEPS = 1000000
SCENARIO = "classic"
ALLOW_BOMB = True
OPPONENT_SETS = [["rule_based_agent", "coin_collector_agent", "peaceful_agent"],]
# REWARDS = {
#     "step": -0.01,
#     "threat": 0.2, ## Hope to make agent more aggressive, by offering an advance on the reward for just trying, NG: delete if interferes
#     e.COIN_COLLECTED: 1.0,
#     e.KILLED_OPPONENT: 5.0,
#     e.CRATE_DESTROYED: 0.1,
#     e.COIN_FOUND: 0.2,
#     e.BOMB_DROPPED: -0.05, ## Agent seems to drop bombs too often, NG: delete if interfers
#     e.GOT_KILLED: -5.0,
# }

REWARDS = {
    "step": -0.01,
    "threat": 0.0, ## Hope to make agent more aggressive, by offering an advance on the reward for just trying, NG: delete if interferes
    "trap": 0.0,
    e.COIN_COLLECTED: 1.5,
    e.KILLED_OPPONENT: 5.0,
    e.CRATE_DESTROYED: 0.5,
    e.COIN_FOUND: 0.2,
    e.BOMB_DROPPED: -0.05,
    e.GOT_KILLED: -5.0,
    "idle_wait": -0.02,
    "backtrack": -0.02,
    "approach": 0.03,
    "empty_bomb": -0.2,
}


LOAD_FROM = "agent_code/ppo_agent/model_v2_classic_nothreat.zip"
TB_NAME = "v2_classic_traprule_idle_3"
N_ENVS = 8
KEYS = ("score", "coins", "crates", "kills", "invalid", "suicide", "died", "bombs", "threats",
        "won", "traps", "waits", "backtracks", "approach", "empty_bombs")
def make_env(rank):
    def _init():
        env = BombermanEnv(
            scenario=SCENARIO,
            opponents=OPPONENT_SETS[rank % len(OPPONENT_SETS)],
            allow_bomb=ALLOW_BOMB, 
            reward_cfg=REWARDS, 
            log_dir=f"logs/ppo_env{rank}"
        )
        return Monitor(env, info_keywords=KEYS)
    return _init

class GameMetrics(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                for k in KEYS:
                    self.logger.record_mean(f"game/{k}", info[k])
        return True
    
if __name__ == "__main__":
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # model = MaskablePPO(
    #     "MlpPolicy", venv,
    #     n_steps=512, batch_size=256, n_epochs=4,
    #     gamma=0.99, gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4,
    #     policy_kwargs=dict(
    #         features_extractor_class=BombermanCNN,
    #         features_extractor_kwargs=dict(features_dim=256),
    #         net_arch=[],
    #     ),
    #     tensorboard_log="runs/ppo", device=DEVICE, verbose=1,
    # )

    model = MaskablePPO.load(LOAD_FROM, env=venv, device=DEVICE,
                            tensorboard_log="runs/ppo", ent_coef=0.01)
    callbacks = [
        GameMetrics(),
        CheckpointCallback(save_freq=2500, save_path="agent_code/ppo_agent/checkpoints",
                           name_prefix=TB_NAME),
    ]
    model.learn(TOTAL_STEPS, tb_log_name=TB_NAME, callback=callbacks,reset_num_timesteps=False)
    model.save("agent_code/ppo_agent/model")
    venv.close()