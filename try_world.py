# import argparse, os
# from environment import BombeRLeWorld

# from agent_code.ppo_agent.features import state_to_planes, legal_actions

# os.makedirs("logs/ppo_try", exist_ok=True)

# args = argparse.Namespace(
#     log_dir="logs/ppo_try", match_name=None, seed=None, scenario="coin-heaven",
#     continue_without_training=True, silence_errors=False,
#     save_replay=False, save_stats=False)


# world = BombeRLeWorld(args, [("user_agent", False)])
# me = world.agents[0]

# world.new_round()
# print("start", me.x, me.y)
# for a in ["RIGHT", "RIGHT", "DOWN", "DOWN", "UP", "LEFT", "LEFT"]:
#     world.do_step(a)
#     state = world.get_state_for_agent(me)
#     print(state_to_planes(state).shape, legal_actions(state, False))
#     print(a, (me.x, me.y), me.events, "score", me.score, "running", world.running)
    
# from agent_code.ppo_agent.env import BombermanEnv

# env = BombermanEnv()
# obs, info = env.reset()
# print(obs.shape)
# for a in [0, 1, 2, 3]:
#     obs, reward, terminated, truncated, info = env.step(a)
#     print(a, env.me.x, env.me.y, env.me.events)


import numpy as np
from gymnasium.utils.env_checker import check_env
from agent_code.ppo_agent.env import BombermanEnv

env = BombermanEnv()
check_env(env)

scores = []
for _ in range(20):
    env.reset()
    done = False
    while not done:
        a = np.random.choice(np.flatnonzero(env.action_masks()))
        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    scores.append(info["score"])
    print(info)
print("mean score", np.mean(scores))
