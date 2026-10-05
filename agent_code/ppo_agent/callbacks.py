import torch
from sb3_contrib import MaskablePPO

from .features import ACTIONS, state_to_planes, legal_actions, trapped_opponents


ALLOW_BOMB = True


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
    torch.set_num_threads(1)
    self.model = MaskablePPO.load("model", device="cpu")
    self.prev_state = None

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if game_state["step"] == 1:
        self.prev_state = None
    obs = state_to_planes(game_state, self.prev_state)
    mask = legal_actions(game_state, ALLOW_BOMB)
    if mask[ACTIONS.index("BOMB")] and trapped_opponents(game_state) > 0:
        self.prev_state = game_state
        return "BOMB"
    action, _ = self.model.predict(obs, action_masks=mask, deterministic=False)
    self.prev_state = game_state
    return ACTIONS[int(action)]



