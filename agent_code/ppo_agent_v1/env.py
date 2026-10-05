import argparse, logging, os
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import events as e
import settings as s
from environment import BombeRLeWorld
from .features import ACTIONS, N_PLANES, blast_timers, state_to_planes, legal_actions


class BombermanEnv(gym.Env):
    def __init__(self, scenario="coin-heaven", opponents=(), allow_bomb=False,
                 reward_cfg=None, log_dir="logs/ppo_env0"):
        os.makedirs(log_dir, exist_ok=True)
        args = argparse.Namespace(
            log_dir=log_dir, match_name=None, seed=None, scenario=scenario,
            continue_without_training=True, silence_errors=False,
            save_replay=False, save_stats=False)
        agents = [("user_agent", False)] + [(o, False) for o in opponents]
        self.world = BombeRLeWorld(args, agents)
        self.world.logger.setLevel(logging.WARNING)
        self.me = self.world.agents[0]

        self.allow_bomb = allow_bomb
        self.reward_cfg = reward_cfg or {"step": -0.01, e.COIN_COLLECTED: 1.0}

        self.observation_space = spaces.Box(0.0, 1.0, (N_PLANES, s.COLS, s.ROWS), np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.world.rng = np.random.default_rng(seed)
        if self.world.running:
            self.world.end_round()
        self.world.new_round()
        self.world.user_input = None
        self.state = self.world.get_state_for_agent(self.me)
        self.obs = state_to_planes(self.state)
        self.counters = dict(coins=0, crates=0, kills=0, invalid=0, suicide=0, died=0, bombs=0, threats=0)
        return self.obs, {}


    def step(self, action):
        before = self.state
        self.world.do_step(ACTIONS[int(action)])
        events = self.me.events

        threatened = 0
        if e.BOMB_DROPPED in events:
            pos = before["self"][3]
            blast = np.isfinite(blast_timers(before["field"], [(pos, s.BOMB_TIMER)]))
            threatened = sum(bool(blast[xy]) for _, _, _, xy in before["others"])

        # self.world.do_step(ACTIONS[int(action)])
        # events = self.me.events
        self.counters["coins"] += events.count(e.COIN_COLLECTED)
        self.counters["crates"] += events.count(e.CRATE_DESTROYED)
        self.counters["kills"] += events.count(e.KILLED_OPPONENT)
        self.counters["invalid"] += events.count(e.INVALID_ACTION)
        self.counters["suicide"] += events.count(e.KILLED_SELF)
        self.counters["died"] += events.count(e.GOT_KILLED)
        self.counters["bombs"] += events.count(e.BOMB_DROPPED)
        self.counters["threats"] += threatened

        reward = self.reward_cfg.get("step", 0.0)
        reward += sum(self.reward_cfg.get(ev, 0.0) for ev in events)
        reward += self.reward_cfg.get("threat", 0.0) * threatened
        dead = self.me.dead
        if dead and self.world.running:
            self.world.end_round()
        done = not self.world.running
        truncated = done and not dead and self.world.step >= s.MAX_STEPS
        terminated = done and not truncated
        state = self.world.get_state_for_agent(self.me)
        #print("step", action, (self.me.x, self.me.y), events, "score", self.me.score, "running", self.world.running, "reward", reward)
        if state is not None:
            self.state = state
            self.obs = state_to_planes(state)
        info = {"score": self.me.score, **self.counters}
        return self.obs, reward, terminated, truncated, info

    def action_masks(self):
        return legal_actions(self.state, self.allow_bomb)
    