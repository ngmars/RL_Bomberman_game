import os
import pickle
import random
from collections import deque
from random import shuffle
import numpy as np

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

#Actions to index 
ACTIONS_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

N_FEATURES = 10
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

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
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.model = np.zeros((len(ACTIONS), N_FEATURES))
        self.epsilon = 0.3
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # Live decision overlay during GUI play (writes live_debug.txt each step)
    self.live_debug = True
    self.live_debug_terminal = False
    if os.path.isfile("live_debug.txt"):
        os.remove("live_debug.txt")

#valid_actions = ACTIONS 
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        #shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile

    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def get_valid_actions(bomb_history, game_state:dict):

    ## EXTRACT KEYS FROM THE DICT
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    # bomb_map = np.ones(arena.shape) * 5
    # for (xb, yb), t in bombs:
    #     for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
    #         if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #             bomb_map[i, j] = min(bomb_map[i, j], t)
    bomb_map = blast_timers(arena, bombs)

    directions = [(x,y), (x+1, y), (x-1, y), (x,y+1), (x,y-1)]
    valid_tiles, valid_actions = [],[]

    for d in directions:
        if( (arena[d] ==0 )and 
            (game_state["explosion_map"][d] < 1) and 
            (bomb_map[d]>0)and 
            (bomb_map[d] == np.inf or bomb_map[x, y] < np.inf) and
            (not d in others) and 
            (not d in bomb_xys)
           ):
            valid_tiles.append(d)
    if(x-1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')

    if(bombs_left>0 and (x,y) not in bomb_history): valid_actions.append("BOMB")
    # self.logger.debug(f'Valid actions: {valid_actions}')
    if not valid_actions: 
        return ['WAIT']
    
    return valid_actions


def _bfs_direction_name(game_state, features):
    _, _, _, (x, y) = game_state['self']
    labels = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    for i, name in enumerate(labels):
        if features[i] > 0.5:
            return name
        if features[5 + i] > 0.5:
            return 'ESCAPE ' + name
    return 'none'


def write_live_debug(self, game_state, valid_actions, q_values, features, action, explored=False):
    if not getattr(self, 'live_debug', False):
        return

    _, score, _, (x, y) = game_state['self']
    lines = [
        f"--- my_agent decision ---",
        f"Step {game_state['step']}  Pos ({x},{y})  Score {score}",
        f"Chosen: {action}" + ("  [explore]" if explored else ""),
        f"Valid:  {', '.join(valid_actions)}",
        f"BFS:    {_bfs_direction_name(game_state, features)}",
        f"Coins:  {len(game_state['coins'])}",
        "Q-values (v=valid, * = chosen):",
    ]
    for a in ACTIONS:
        valid_mark = "v" if a in valid_actions else " "
        chosen_mark = "*" if a == action else " "
        q = q_values[ACTIONS_TO_IDX[a]]
        lines.append(f" {valid_mark}{chosen_mark} {a:5s} {q:8.2f}")

    with open("live_debug.txt", "w") as f:
        f.write("\n".join(lines))

    if getattr(self, 'live_debug_terminal', False):
        print("\n".join(lines), flush=True)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #valid_actions = get_valid_actions(game_state)
    valid_actions = get_valid_actions(self.bomb_history, game_state)
    features = state_to_features(game_state)
    q_values = self.model @ features

    # Explore
    if self.train and random.random() < self.epsilon:
        action = random.choice(valid_actions)
        write_live_debug(self, game_state, valid_actions, q_values, features, action, explored=True)
        return action

    best_action = None
    best_q = -np.inf

    for a in valid_actions:
        if q_values[ACTIONS_TO_IDX[a]] > best_q:
            best_q = q_values[ACTIONS_TO_IDX[a]]
            best_action = a

    if best_action == 'BOMB':
        _, _, _, (x, y) = game_state['self']
        self.bomb_history.append((x, y))

    action = best_action if best_action is not None else 'WAIT'
    write_live_debug(self, game_state, valid_actions, q_values, features, action, explored=False)
    return action




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

    game state: 
    round':
    'step': 
    'field':
    'self': 
    'others': 
    'bombs':
    'coins': 
    'user_input':
    'explosion_map':
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    field = game_state["field"]
    agent_name, agent_score, bombs_left, (x, y) = game_state["self"]
    coins = game_state["coins"]
    explosion_map = game_state["explosion_map"]
    max_dist = max(s.COLS, s.ROWS)
    danger = blast_timers(field, game_state["bombs"])
    free_space = (field == 0) & (explosion_map < 1)
    for (_, _, _, other_xy) in game_state["others"]:
        free_space[other_xy] = False
    for (bomb_xy, _) in game_state["bombs"]:
        free_space[bomb_xy] = False

    start = (x, y)
    in_danger = danger[x, y] < np.inf

    if in_danger:
        # head for the nearest tile no bomb can reach
        targets = [tuple(p) for p in np.argwhere(free_space & (danger == np.inf))]
        first_step = look_for_targets(free_space, start, targets)
    else:
        # head for the nearest coin without walking through blast zones
        free_space = free_space & (danger == np.inf)
        first_step = look_for_targets(free_space, start, coins)

    # Order: UP, RIGHT, DOWN, LEFT
    direction = [0.0, 0.0, 0.0, 0.0]
    if first_step == (x, y - 1):
        direction[0] = 1.0
    elif first_step == (x + 1, y):
        direction[1] = 1.0
    elif first_step == (x, y + 1):
        direction[2] = 1.0
    elif first_step == (x - 1, y):
        direction[3] = 1.0
    block = direction + [0.0 if any(direction) else 1.0]

    # features 0-4: safe (coin direction, none); 5-9: in danger (escape direction, none)
    features = [0.0] * 5 + block if in_danger else block + [0.0] * 5
    return np.array(features, dtype=np.float32)
    # free_space = (field == 0) & (explosion_map < 1)
    # for (_, _, _, other_xy) in game_state["others"]:
    #     free_space[other_xy] = False


    # ## find the nearest coin location
    # # if coins: 
    # #     nearest = min(coins, key = lambda c:abs(c[0]-x)+ abs(c[1]-y))
    # #     cx, cy = nearest 
    # #     features += [(cx-x)/max_dist, (cy-y)/max_dist, (abs(cx-x)+ abs(cy-y))/max_dist]
    # # else:
    # #     features += [0.0,0.0,0.0]

    # start = (x,y)

    # if coins:
    #     first_step = look_for_targets(free_space, start, coins)
    # else:
    #     first_step  = None

    # # Order must match  walkability loop: UP, RIGHT, DOWN, LEFT
    # bfs_features = [0.0, 0.0, 0.0, 0.0]

    # if first_step == (x, y - 1):
    #     bfs_features[0] = 1.0   # UP
    # elif first_step == (x + 1, y):
    #     bfs_features[1] = 1.0   # RIGHT
    # elif first_step == (x, y + 1):
    #     bfs_features[2] = 1.0   # DOWN
    # elif first_step == (x - 1, y):
    #     bfs_features[3] = 1.0   # LEFT
    # # else: all zeros (already on coin, no coins, or stuck)

    # #features.extend(bfs_features)
    # ## check where agent can move
    # ## UP, RIGHT, DOWN, LEFT
    # # for dx, dy in ([(0,-1),(1,0),(0,1),(-1,0)]):
    # #     nx, ny = x+dx, y+dy
    # #     if (0<= nx < field.shape[0] and 0 <= ny < field.shape[1]):
    # #         ## if nx,ny is walkable and in the field and is not in the current explosion zone
    # #         walkable =  field[nx, ny] ==0 and explosion_map[nx,ny]<1
    # #         if(walkable):
    # #             features.append(1.0)
    # #         else:
    # #             features.append(0.0)
    # #     else:
    # #         features.append(0.0)

    # # ## check if any bombs are left on the field
    # # features.append(1.0 if bombs_left else 0.0)
    # # return np.array(features, dtype=np.float32)
    # features = bfs_features + [0.0 if any(bfs_features) else 1.0]
    # #features = bfs_features + [1.0]   # 4 BFS one-hots + bias
    # return np.array(features, dtype=np.float32)

    # # For example, you could construct several channels of equal shape, ...   
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)
