# keep this line for cython directives
class MCTSMeta:
    EXPLORATION = 0.8
    RAVE_CONST = 300
    RANDOMNESS = 0.5
    K_CONST = 10
    A_CONST = 0.25

class GameMeta:
    PLAYERS = {'none': 0, 'red': 1, 'blue': 2}
    INF = float('inf')
    GAME_OVER = -1
    EDGE1 = (12, 1)
    EDGE2 = (12, 2)
    NEIGHBOR_PATTERNS = ((-1, 0), (1, 0),(0, -1), (0, 1), (-1, 1),(1, -1))
    
