import numpy as np

import settings as s

WALLS, CRATES, COINS, ME, OTHERS, BOMBS, DANGER, EXPLOSION, CAN_BOMB = range(9)

# number of channels for network input
N_PLANES = 9

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

## Create a mask that will help agent make only safe moves
SAFE_MASK = True
BLOCK_OTHERS = True
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER
BOMB_SLACK = True
SLACK_RANGE = 4

def lethal_schedule(field, bombs, explosion_map):
    lethal = np.zeros((HORIZON, *field.shape), dtype=bool)
    for k in range(HORIZON):
        lethal[k] = explosion_map > k
    for bomb in bombs:
        hit = np.isfinite(blast_timers(field, [bomb]))
        t = bomb[1]
        lethal[t:t + s.EXPLOSION_TIMER][:, hit] = True
    return lethal

def can_survive(field, bomb_tiles, lethal, pos):
    frontier = {pos}
    for k in range(1, HORIZON):
        reachable = set()
        for (x, y) in frontier:
            for dx, dy in DIRS + [(0, 0)]:
                tile = (x + dx, y + dy)
                if (dx or dy) and (field[tile] != 0 or tile in bomb_tiles):
                    continue
                if not lethal[k][tile]:
                    reachable.add(tile)
        if not reachable:
            return False
        frontier = reachable
    return True


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
        9. Bombs (where bombs are located)
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

    for (bx, by), _ in game_state["bombs"]:
        planes[BOMBS, bx, by] = 1

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
    assert planes[BOMBS].sum() == len(game_state["bombs"])

def legal_actions(game_state, allow_bomb):
    field = game_state['field']
    _, _, can_bomb, (x, y) = game_state["self"]
    blocked = {xy for xy, _ in game_state["bombs"]}
    blocked |= {xy for _, _, _, xy in game_state["others"]}
    timers = blast_timers(field, game_state["bombs"])
    lethal = (game_state["explosion_map"] > 0) | (timers == 0)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    for i, (dx, dy) in enumerate(DIRS):
        tile = (x + dx, y + dy)
        mask[i] = field[tile] == 0 and tile not in blocked and not lethal[tile]


    stay_ok = not (lethal[x, y] and mask[:4].any()) #false when current tile will explode, but there is a safe spot near
    mask[ACTIONS.index("WAIT")] = stay_ok
    mask[ACTIONS.index("BOMB")] = stay_ok and can_bomb and allow_bomb

    ## START OF SAFE MASK CODE : NG Delete if  no improvement
    # testing code, remove if bad performance: it's to add 
    # A safe mask, so that the agent doesn't go into a bomb zone
    ##
    if not SAFE_MASK:
        return mask

    bombs = game_state["bombs"]
    bomb_tiles = {xy for xy, _ in bombs}
    obstacles = blocked if BLOCK_OTHERS else bomb_tiles
    schedule = lethal_schedule(field, bombs, game_state["explosion_map"])
    safe = np.zeros(len(ACTIONS), dtype=bool)

    for i, (dx, dy) in enumerate(DIRS):
        tile = (x + dx, y + dy)
        safe[i] = mask[i] and can_survive(field, obstacles, schedule, tile)
    if not lethal[x, y]:
        safe[ACTIONS.index("WAIT")] = can_survive(field, obstacles, schedule, (x, y))
        if mask[ACTIONS.index("BOMB")]:
            new_bombs = [((x, y), s.BOMB_TIMER)]
            near = any(abs(ox - x) + abs(oy - y) <= SLACK_RANGE for _, _, _, (ox, oy) in game_state["others"])
            if BOMB_SLACK and near:
                new_bombs.append(((x, y), s.BOMB_TIMER - 1))
            with_bomb = lethal_schedule(field, bombs + new_bombs, game_state["explosion_map"])
            safe[ACTIONS.index("BOMB")] = can_survive(field, obstacles | {(x, y)}, with_bomb, (x, y))
    return safe if safe.any() else mask

    ### END OF SAFE MASK