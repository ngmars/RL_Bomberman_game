import argparse, logging, os
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import events as e
import settings as s
from environment import BombeRLeWorld
from .features import ACTIONS, N_PLANES, blast_timers, distance_to_goal, state_to_planes, legal_actions, trapped_opponents, trapped_opponents


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
        self.last_pos = None
        self.obs = state_to_planes(self.state)
        self.counters = dict(coins=0, crates=0, kills=0, invalid=0, suicide=0, died=0, bombs=0,
                             threats=0, won=0, traps=0, waits=0, backtracks=0,
                             approach=0, empty_bombs=0)
        return self.obs, {}


    def step(self, action):
        before = self.state
        self.world.do_step(ACTIONS[int(action)])
        events = self.me.events
        pos_before = tuple(before["self"][3])
        pos_after = (self.me.x, self.me.y)
        backtrack = int(pos_after != pos_before and pos_after == self.last_pos)
        self.last_pos = pos_before
        idle_wait = int(e.WAITED in events and before["self"][2])
        approach = 0
        in_danger = np.isfinite(blast_timers(before["field"], before["bombs"]))[pos_before] \
            or before["explosion_map"][pos_before] > 0
        goal_map = None if in_danger else distance_to_goal(before)
        if goal_map is not None and not self.me.dead:
            approach = int(np.sign(goal_map[pos_before] - goal_map[pos_after]))
        empty_bomb = 0
        threatened = 0
        trapped = 0
        if e.BOMB_DROPPED in events:
            pos = before["self"][3]
            blast = np.isfinite(blast_timers(before["field"], [(pos, s.BOMB_TIMER)]))
            threatened = sum(bool(blast[xy]) for _, _, _, xy in before["others"])
            trapped = trapped_opponents(before)

            #penalize dropping bombs that don't hit crates or opponents
            crates_hit = int(((before["field"] == 1) & blast).sum())
            empty_bomb = int(crates_hit == 0 and threatened == 0)

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
        self.counters["traps"] += trapped
        self.counters["waits"] += idle_wait
        self.counters["backtracks"] += backtrack    
        self.counters["approach"] += approach
        self.counters["empty_bombs"] += empty_bomb

        reward = self.reward_cfg.get("step", 0.0)
        reward += sum(self.reward_cfg.get(ev, 0.0) for ev in events)
        reward += self.reward_cfg.get("threat", 0.0) * threatened
        reward += self.reward_cfg.get("trap", 0.0) * trapped
        reward += self.reward_cfg.get("idle_wait", 0.0) * idle_wait
        reward += self.reward_cfg.get("backtrack", 0.0) * backtrack
        reward += self.reward_cfg.get("approach", 0.0) * approach
        reward += self.reward_cfg.get("empty_bomb", 0.0) * empty_bomb
        dead = self.me.dead
        ## WIN COUNTER REWARD
        if dead:
            while self.world.running:
                self.world.do_step("WAIT")
        done = not self.world.running
        if done and len(self.world.agents) > 1:
            best_other = max(a.score for a in self.world.agents[1:])
            result = (self.me.score > best_other) - (self.me.score < best_other)
            self.counters["won"] = int(result > 0)
            reward += self.reward_cfg.get("win", 0.0) * result

        truncated = done and not dead and self.world.step >= s.MAX_STEPS
        terminated = done and not truncated
        state = self.world.get_state_for_agent(self.me)
        #print("step", action, (self.me.x, self.me.y), events, "score", self.me.score, "running", self.world.running, "reward", reward)
        if state is not None:
            self.state = state
            self.obs = state_to_planes(state, before)
        info = {"score": self.me.score, **self.counters}
        return self.obs, reward, terminated, truncated, info

    def action_masks(self):
        return legal_actions(self.state, self.allow_bomb)
    