import numpy as np
import torch
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO

from agent_code.ppo_agent.env import BombermanEnv
from agent_code.ppo_agent.features import ACTIONS, blast_timers

MODEL = "agent_code/ppo_agent/model"
N_ROUNDS = 40
MODES = ["sampled", "greedy_in_danger", "greedy"]

def action_probs(model, obs, mask):
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask)
    p = dist.distribution.probs[0].cpu().numpy().astype(np.float64)
    return p / p.sum()

def in_danger(state):
    x, y = state["self"][3]
    return np.isfinite(blast_timers(state["field"], state["bombs"])[x, y])


def play(model, env, mode):
    steps, rounds = [], []
    for _ in range(N_ROUNDS):
        env.reset()
        done, bombs = False, 0
        while not done:
            p = action_probs(model, env.obs, env.action_masks())
            danger = in_danger(env.state)
            greedy = mode == "greedy" or (mode == "greedy_in_danger" and danger)
            a = int(p.argmax()) if greedy else int(np.random.choice(len(p), p=p))
            steps.append((danger, p.max()))
            bombs += ACTIONS[a] == "BOMB"
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
        rounds.append((env.world.step, info["coins"], info["crates"], bombs, env.me.dead))
    return np.array(steps), np.array(rounds, dtype=float)


if __name__ == "__main__":
    model = MaskablePPO.load(MODEL, device="cpu")
    env = BombermanEnv(scenario="loot-crate", allow_bomb=True, log_dir="logs/ppo_analyze")
    results = {m: play(model, env, m) for m in MODES}

    print(f"{'mode':18s} {'died':>6s} {'steps':>7s} {'coins':>6s} {'crates':>7s} {'bombs':>6s}")
    for m, (_, r) in results.items():
        print(f"{m:18s} {r[:, 4].mean():6.2f} {r[:, 0].mean():7.1f} {r[:, 1].mean():6.1f} "
              f"{r[:, 2].mean():7.1f} {r[:, 3].mean():6.1f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(MODES, [results[m][1][:, 4].mean() for m in MODES])
    ax[0].set_title("Share of rounds ending in death")
    ax[0].set_ylim(0, 1)
    ax[1].bar(MODES, [results[m][1][:, 3].mean() for m in MODES])
    ax[1].set_title("Bombs dropped per round")
    s = results["sampled"][0]
    bins = np.linspace(0, 1, 21)
    ax[2].hist(s[s[:, 0] == 1, 1], bins, alpha=0.6, density=True, label="in a blast zone")
    ax[2].hist(s[s[:, 0] == 0, 1], bins, alpha=0.6, density=True, label="safe")
    ax[2].set_title("Probability of the policy's top move")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig("agent_code/ppo_agent/escape_analysis.png", dpi=120)
    print("saved agent_code/ppo_agent/escape_analysis.png")