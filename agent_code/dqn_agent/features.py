import numpy as np

import settings as s

WALLS, CRATES, COINS, ME, OTHERS, DANGER, EXPLOSION, CAN_BOMB = range(8)

# number of channels for network input
N_PLANES = 8

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

def blast_timers(field, bombs):
    # when a tile will be hit by the bomb, by default np.inf, if no bomb on the tile
    timers = np.full(field.shape, np.inf)
    for (bx, by), t in bombs:
        timers[bx, by] = min(timers[bx, by], t)
        for dx, dy in DIRS:
            for k in range(1, s.BOMB_POWER + 1):
                i, j = bx + dx * k, by + dy * k
                if field[i, j] == -1:
                    break
                timers[i, j] = min(timers[i, j], t)
    return timers

def state_to_planes(game_state):
    """
    Converts the game state to a set of planes, tells the model where the
        1. walls
        2. crates
        3. coins
        4. my agent
        5. other agents 
        6. Danger (where bombs will explode)
        7. Explosion (where bombs are exploding)
        8. Can bomb (if the agent can bomb)
    are located in the game.
    """
    if game_state is None:
        return None

    ## Set up the planes
    field = game_state['field']
    _, _, can_bomb, (x, y) = game_state["self"]

    planes = np.zeros((N_PLANES, *field.shape), dtype=np.float32)

    planes[WALLS] = field == -1
    planes[CRATES] = field == 1

    for cx, cy in game_state["coins"]:
        planes[COINS, cx, cy] = 1

    planes[ME, x, y] = 1

    for _, _, _, (ox, oy) in game_state["others"]:
        planes[OTHERS, ox, oy] = 1

    timers = blast_timers(field, game_state["bombs"])
    # mark the tiles where the bomb effect will be there
    hit = np.isfinite(timers)
    planes[DANGER][hit] = (s.BOMB_TIMER - timers[hit]) / s.BOMB_TIMER
    planes[EXPLOSION] = game_state["explosion_map"] > 0
    planes[CAN_BOMB] = float(can_bomb)
    #print(planes)
    return planes

def check_planes(planes, game_state):
    ## Ensure that the planes are always holding sane values, if this 
    ## assumption fails, model isn't learning the right thing
    _, _, can_bomb, (x, y) = game_state["self"]
    assert planes.shape == (N_PLANES, s.COLS, s.ROWS)
    assert planes.dtype == np.float32
    assert planes.min() >= 0 and planes.max() <= 1
    assert planes[WALLS].sum() == 113
    assert planes[ME].sum() == 1 and planes[ME, x, y] == 1
    assert planes[COINS].sum() == len(game_state["coins"])
    assert planes[OTHERS].sum() == len(game_state["others"])
    assert (planes[WALLS] * planes[CRATES]).sum() == 0


