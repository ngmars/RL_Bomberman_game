from collections import namedtuple, deque
import numpy as np
import pickle
from typing import List
import os
import csv
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#Actions to index 
ACTIONS_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# # Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# # Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"

METRICS_HEADER = [
    "round", "game_score", "step", "episode_reward", "epsilon",
    "max_q", "q_chosen", "q_up", "q_right", "q_down", "q_left", "q_wait", "q_bomb",
]


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # # Example: Setup an array that will note transition tuples
    # # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.transitions = deque(maxlen=10000)
    self.alpha = 0.1  # lr
    self.gamma = 0.99 # discount    
    self.episode_reward = 0
    self.metrics_path = "training_metrics.csv"
    # Delete training_metrics.csv before a new run to start fresh.
    if not os.path.isfile(self.metrics_path):
        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)

def distance_to_nearest_coint(game_state):
    if(game_state is None):
        return None

    _,_, _, (x,y) = game_state['self']
    coins = game_state['coins']

    if not coins:
        return None
    return min(abs(cx-x) + abs(cy-y) for cx,cy in coins)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # # Idea: Add your own events to hand out rewards
    old_dist = distance_to_nearest_coint(old_game_state)
    new_dist = distance_to_nearest_coint(new_game_state)

    if old_dist is not None and new_dist is not None:
        if(new_dist < old_dist):
            events.append("MOVED_TOWARD_COIN")
        # elif(new_dist > old_dist):
        #     events.append("MOVED_AWAY_FROM_COIN")

    reward = reward_from_events(self, events)
    self.episode_reward += reward  
    done = False

    self.transitions.append(Transition(
        state_to_features(old_game_state),
        ACTIONS_TO_IDX[self_action],
        state_to_features(new_game_state),
        reward,
        done
    ))
    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

def train_on_batch(self):
    for t in self.transitions:
        s, a, s_next, r, done = t.state, t.action, t.next_state, t.reward, t.done
        if s is None:
            continue
        q_predicted = self.model[a] @ s

        if done or s_next is None:
            q_target = r

        else:
            q_target = r + self.gamma * np.max(self.model @ s_next)

        td_error = q_target - q_predicted
        self.model[a] += self.alpha * td_error * s # this is the gradient of the linear Q
    self.transitions.clear()


def log_q_values(self, game_state, action_taken=None):
    features = state_to_features(game_state)
    if features is None:
        return {key: 0.0 for key in METRICS_HEADER if key.startswith("q_") or key == "max_q"}

    q_values = self.model @ features
    row = {
        "max_q": float(np.max(q_values)),
        "q_up": float(q_values[ACTIONS_TO_IDX['UP']]),
        "q_right": float(q_values[ACTIONS_TO_IDX['RIGHT']]),
        "q_down": float(q_values[ACTIONS_TO_IDX['DOWN']]),
        "q_left": float(q_values[ACTIONS_TO_IDX['LEFT']]),
        "q_wait": float(q_values[ACTIONS_TO_IDX['WAIT']]),
        "q_bomb": float(q_values[ACTIONS_TO_IDX['BOMB']]),
        "q_chosen": 0.0,
    }
    if action_taken is not None:
        row["q_chosen"] = float(q_values[ACTIONS_TO_IDX[action_taken]])
    return row


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    final_reward = reward_from_events(self, events)
    self.episode_reward += final_reward
    self.transitions.append(Transition(
        state_to_features(last_game_state),
        ACTIONS_TO_IDX[last_action],
        None,
        final_reward,
        True
    ))

    train_on_batch(self)
    self.epsilon = max(0.05, self.epsilon * 0.995)
    q_stats = log_q_values(self, last_game_state, last_action)

    _, game_score, _, _ = last_game_state["self"]
    with open(self.metrics_path, "a", newline="") as f:
        csv.writer(f).writerow([
            last_game_state["round"],
            game_score,
            last_game_state["step"],
            self.episode_reward,
            self.epsilon,
            q_stats["max_q"],
            q_stats["q_chosen"],
            q_stats["q_up"],
            q_stats["q_right"],
            q_stats["q_down"],
            q_stats["q_left"],
            q_stats["q_wait"],
            q_stats["q_bomb"],
        ])
    self.episode_reward = 0

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -5,
        e.KILLED_SELF: -50,
        e.SURVIVED_ROUND: 5,
        "MOVED_TOWARD_COIN": 0.1,
        "MOVED_AWAY_FROM_COIN": -0.1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
