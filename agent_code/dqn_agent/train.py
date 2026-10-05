import csv
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F

import events as e
from .callbacks import ACTIONS, MODEL_FILE, legal_actions
from .features import state_to_planes
from .model import QNetwork


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "next_mask", "done"))

# Hyper parameters -- DO modify
GAMMA = 0.97
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
WARMUP_STEPS = 2_000
TRAIN_EVERY = 4
TARGET_SYNC_EVERY = 2_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 100_000

# Events
STEP_COST = -0.01
GAME_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.GOT_KILLED: -5.0,
}

TRAIN_STATE_FILE = "train_state.pt"
METRICS_FILE = "training_metrics.csv"
METRICS_HEADER = ["round", "score", "steps", "reward", "epsilon", "loss", "mean_q"]


CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 25


def epsilon_at(step):
    fraction = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + fraction * (EPS_END - EPS_START)


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    ## Set up Mac MPS or CUDA - need GPU for training
    if torch.backends.mps.is_available():
        self.device = torch.device("mps")
    elif torch.cuda.is_available():
        self.device = torch.device("cuda")
    self.model.to(self.device)


    self.target_model = QNetwork().to(self.device)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.buffer = deque(maxlen=BUFFER_SIZE)
    self.total_steps = 0

    #in case training crashed, need both the model.pt file and train_State.pt file to resume training
    #from same checkpoint, else we start from scratch
    if os.path.isfile(MODEL_FILE) and os.path.isfile(TRAIN_STATE_FILE):
        saved = torch.load(TRAIN_STATE_FILE, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.total_steps = saved["total_steps"]
        self.logger.info(f"Resuming training at step {self.total_steps}.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    self.epsilon = epsilon_at(self.total_steps)
    self.round_reward = 0.0
    self.round_losses = []
    self.round_qs = []

    if not os.path.isfile(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_HEADER)


def reward_from_events(events):
    return STEP_COST + sum(GAME_REWARDS.get(event, 0.0) for event in events)


def optimize(self):
    batch = random.sample(self.buffer, BATCH_SIZE)
    states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device)
    actions = torch.tensor([t.action for t in batch], device=self.device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
    next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device)
    next_masks = torch.from_numpy(np.stack([t.next_mask for t in batch])).to(self.device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

    q_taken = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online = self.model(next_states).masked_fill(~next_masks, -torch.inf)
        best_next = next_q_online.argmax(dim=1, keepdim=True)
        next_q = self.target_model(next_states).gather(1, best_next).squeeze(1)
        targets = rewards + GAMMA * next_q * (1 - dones)

    loss = F.smooth_l1_loss(q_taken, targets)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
    self.optimizer.step()

    self.round_losses.append(loss.item())
    self.round_qs.append(q_taken.mean().item())


def remember(self, state, action, next_state, events, done):
    reward = reward_from_events(events)
    self.round_reward += reward

    planes = state_to_planes(state)
    if done:
        next_planes = np.zeros_like(planes)
        next_mask = np.ones(len(ACTIONS), dtype=bool)
    else:
        next_planes = state_to_planes(next_state)
        next_mask = legal_actions(next_state)
    self.buffer.append(Transition(
        planes, ACTIONS.index(action), reward, next_planes, next_mask, done))

    self.total_steps += 1
    self.epsilon = epsilon_at(self.total_steps)
    if len(self.buffer) >= WARMUP_STEPS and self.total_steps % TRAIN_EVERY == 0:
        optimize(self)
    if self.total_steps % TARGET_SYNC_EVERY == 0:
        self.target_model.load_state_dict(self.model.state_dict())
    
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
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
    remember(self, old_game_state, self_action, new_game_state, events, done=False)


def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    if e.GOT_KILLED in events:
        remember(self, last_game_state, last_action, None, events, done=True)

    torch.save(self.model.state_dict(), MODEL_FILE)
    torch.save({"optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps}, TRAIN_STATE_FILE)
    
    if last_game_state["round"] % CHECKPOINT_EVERY == 0:
        path = os.path.join(CHECKPOINT_DIR, f"step_{self.total_steps}.pt")
        torch.save(self.model.state_dict(), path)
        
    with open(METRICS_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            last_game_state["round"],
            last_game_state["self"][1],
            last_game_state["step"],
            round(self.round_reward, 2),
            round(self.epsilon, 3),
            np.mean(self.round_losses) if self.round_losses else "",
            np.mean(self.round_qs) if self.round_qs else "",
        ])
    self.round_reward = 0.0
    self.round_losses = []
    self.round_qs = []


# def reward_from_events(self, events: List[str]) -> int:
#     """
#     *This is not a required function, but an idea to structure your code.*

#     Here you can modify the rewards your agent get so as to en/discourage
#     certain behavior.
#     """
#     game_rewards = {
#         e.COIN_COLLECTED: 1,
#         e.KILLED_OPPONENT: 5,
#         PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
#     }
#     reward_sum = 0
#     for event in events:
#         if event in game_rewards:
#             reward_sum += game_rewards[event]
#     self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
#     return reward_sum
