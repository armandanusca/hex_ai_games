# keep this line for cython directives

cdef class GameState:
    """
    Stores information representing the current state of a game of hex, namely
    the board and the current turn. Also provides functions for playing game.
    """
    # dictionary associating numbers with players
    # PLAYERS = {"none": 0, "red": 1, "blue": 2}

    # move value of -1 indicates the game has ended so no move is possible
    # GAME_OVER = -1

    # represent edges in the union find structure for detecting the connection
    # for player 1 Edge1 is high and EDGE2 is low
    # for player 2 Edge1 is left and EDGE2 is right

    # neighbor_patterns = ((-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))
    cdef public:
        int size
        int to_play
        object board
        int red_played
        int blue_played
        object red_groups
        object blue_groups

    cpdef void play(self, tuple cell)
    cpdef dict get_num_played(self)

    cpdef void place_red(self, tuple cell)
    cpdef void place_blue(self, tuple cell)
    cpdef int turn(self)
    cpdef void set_turn(self, int player)
    cpdef int winner(self)
    cpdef list neighbors(self, tuple cell)
    cpdef list moves(self)
    cpdef tuple get_rb_played(self)