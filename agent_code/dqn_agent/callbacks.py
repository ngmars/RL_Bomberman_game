import os
import random

import numpy as np
import torch

from .features import state_to_planes, DIRS
from .model import QNetwork

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = os.environ.get("DQN_WEIGHTS", "model.pt")
ALLOW_BOMB = False


def blast_timers(field, bombs):
    """Per tile: steps until a bomb blast hits it (np.inf if no bomb reaches it)."""
    danger = np.full(field.shape, np.inf)
    for (bx, by), t in bombs:
        danger[bx, by] = min(danger[bx, by], t)
        for dx, dy in DIRS:
            for k in range(1, s.BOMB_POWER + 1):
                i, j = bx + dx * k, by + dy * k
                if field[i, j] == -1:
                    break
                danger[i, j] = min(danger[i, j], t)
    return danger


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # if self.train or not os.path.isfile("my-saved-model.pt"):
    #     self.logger.info("Setting up model from scratch.")
    #     weights = np.random.rand(len(ACTIONS))
    #     self.model = weights / weights.sum()
    # else:
    #     self.logger.info("Loading model from saved state.")
    #     with open("my-saved-model.pt", "rb") as file:
    #         self.model = pickle.load(file)

    torch.set_num_threads(1)
    self.model = QNetwork()
    if "DQN_WEIGHTS" in os.environ and not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"DQN_WEIGHTS points to a missing file: {MODEL_FILE}")
    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading model from saved state.")
        weights = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
        if os.path.isfile(MODEL_FILE):
            weights = torch.load(MODEL_FILE, map_location=self.device, weights_only=True)
            self.model.load_state_dict(weights)
            self.logger.info("Loaded weights from file.")
        else:
            self.logger.info("No weights file, starting with random weights.")
        self.device = torch.device("cpu")
        self.model.eval()
        self.epsilon = 0.0

def legal_actions(game_state: dict):
    field = game_state['field']
    _, _, can_bomb, (x, y) = game_state["self"]
    blocked = {xy for xy, _ in game_state["bombs"]}
    blocked |= {xy for _, _, _, xy in game_state["others"]}

    mask = np.zeros(len(ACTIONS), dtype=bool)
    for i, (dx, dy) in enumerate(DIRS):
        tile = (x + dx, y + dy)
        mask[i] = field[tile] == 0 and tile not in blocked

    mask[ACTIONS.index("WAIT")] = True
    mask[ACTIONS.index('BOMB')] = can_bomb and ALLOW_BOMB

    return mask


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # check_planes(state_to_planes(game_state), game_state)
    # random_prob = .1
    # if self.train and random.random() < random_prob:
    #     self.logger.debug("Choosing action purely at random.")
    #     # 80%: walk in any direction. 10% wait. 10% bomb.
    #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # self.logger.debug("Querying model for action.")
    # return np.random.choice(ACTIONS, p=self.model)

    mask = legal_actions(game_state)
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        return ACTIONS[random.choice(np.flatnonzero(mask))]

    planes = torch.from_numpy(state_to_planes(game_state)).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.model(planes)[0].cpu().numpy()
    q_values[~mask] = -np.inf
    return ACTIONS[int(np.argmax(q_values))]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
