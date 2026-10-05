import argparse, csv, logging, os

import numpy as np

from environment import BombeRLeWorld
import agent_code.ppo_agent.features as F

N_ROUNDS = 300
OUT_DIR = "results/report"
RULE_3 = ["rule_based_agent"] * 3
RULE_1 = ["rule_based_agent"]


CONFIGS = [
    ("4p_full", "ppo_agent", RULE_3, {}),
    ("4p_no_coin_rule", "ppo_agent", RULE_3, {"COIN_RULE": False}),
    ("4p_no_trap_rule", "ppo_agent", RULE_3, {"TRAP_RULE": False}),
    ("4p_no_rules", "ppo_agent", RULE_3, {"COIN_RULE": False, "TRAP_RULE": False}),
    ("4p_v1", "ppo_agent_v1", RULE_3, {}),
    ("4p_vs_collectors", "ppo_agent", ["coin_collector_agent"] * 3, {}),
    ("duel_full", "ppo_agent", RULE_1, {}),
    ("duel_no_coin_rule", "ppo_agent", RULE_1, {"COIN_RULE": False}),
    ("duel_no_trap_rule", "ppo_agent", RULE_1, {"TRAP_RULE": False}),
    ("duel_no_rules", "ppo_agent", RULE_1, {"COIN_RULE": False, "TRAP_RULE": False}),
    ("duel_v1", "ppo_agent_v1", RULE_1, {}),
]
METRICS = ["win", "draw", "score", "coins", "kills", "suicide", "died", "bombs", "crates",
           "alive_steps", "opp_score", "opp_coins", "opp_kills", "opp_suicide"]


def play(name, agent, opponents, switches):
    defaults = {k: getattr(F, k) for k in switches}
    for k, v in switches.items():
        setattr(F, k, v)

    args = argparse.Namespace(
        log_dir=OUT_DIR, match_name=None, seed=None, scenario="classic",
        continue_without_training=True, silence_errors=False,
        save_replay=False, save_stats=False)
    world = BombeRLeWorld(args, [(agent, False)] + [(o, False) for o in opponents])
    world.logger.setLevel(logging.WARNING)
    me, others = world.agents[0], world.agents[1:]
    rows = []
    for r in range(N_ROUNDS):
        world.new_round()
        while world.running:
            world.do_step("WAIT")
        best = max(o.score for o in others)
        rows.append(dict(
            config=name, round=r + 1, steps=world.step,
            win=int(me.score > best), draw=int(me.score == best), score=me.score,
            coins=me.statistics["coins"], kills=me.statistics["kills"],
            suicide=me.statistics["suicides"], died=int(me.dead),
            bombs=me.statistics["bombs"], crates=me.statistics["crates"],
            alive_steps=me.statistics["steps"],
            opp_score=np.mean([o.score for o in others]),
            opp_coins=np.mean([o.statistics["coins"] for o in others]),
            opp_kills=np.mean([o.statistics["kills"] for o in others]),
            opp_suicide=np.mean([o.statistics["suicides"] for o in others]),
        ))

    for k, v in defaults.items():
        setattr(F, k, v)
    return rows


def summarise(rows):
    out = {"config": rows[0]["config"], "rounds": len(rows)}
    for m in METRICS:
        values = np.array([r[m] for r in rows], dtype=float)
        out[m] = values.mean()
        out[m + "_ci"] = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows, summaries = [], []
    for name, agent, opponents, switches in CONFIGS:
        rows = play(name, agent, opponents, switches)
        all_rows += rows
        s = summarise(rows)
        summaries.append(s)
        print(f"{name:22s} win {s['win']:.2f}±{s['win_ci']:.2f}  score {s['score']:.2f}±{s['score_ci']:.2f}  "
              f"coins {s['coins']:.2f}  kills {s['kills']:.2f}  died {s['died']:.2f}  "
              f"opp score {s['opp_score']:.2f}", flush=True)

    with open(f"{OUT_DIR}/rounds.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(f"{OUT_DIR}/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print("saved", f"{OUT_DIR}/rounds.csv", "and", f"{OUT_DIR}/summary.csv")