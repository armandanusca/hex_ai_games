from numpy import zeros, int_
from unionfind cimport UnionFind
from meta import GameMeta
from copy import deepcopy


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

    def __init__(self, size):
        """
        Initialize the game board and give red first turn.
        Also create our union find structures for win checking.
        Args:
            size (int): The board size
        """
        self.size = size
        self.to_play = GameMeta.PLAYERS['red']
        self.board = zeros((size, size))
        self.board = int_(self.board)
        self.red_played = 0
        self.blue_played = 0
        self.red_groups = UnionFind()
        self.blue_groups = UnionFind()
        self.red_groups.set_ignored_elements([GameMeta.EDGE1, GameMeta.EDGE2])
        self.blue_groups.set_ignored_elements([GameMeta.EDGE1, GameMeta.EDGE2])

    def __deepcopy__(self, memo):
        """
        Deep copy of the game state.
        """
        new_state = GameState(self.size)
        new_state.to_play = self.to_play
        new_state.board = self.board.copy()
        new_state.red_played = self.red_played
        new_state.blue_played = self.blue_played
        new_state.red_groups = deepcopy(self.red_groups)
        new_state.blue_groups = deepcopy(self.blue_groups)
        return new_state

    cpdef void play(self, tuple cell):
        """
        Play a stone of the player that owns the current turn in input cell.
        Args:
            cell (tuple): row and column of the cell
        """
        if self.to_play == GameMeta.PLAYERS['red']:
            self.place_red(cell)
            self.to_play = GameMeta.PLAYERS['blue']
        elif self.to_play == GameMeta.PLAYERS['blue']:
            self.place_blue(cell)
            self.to_play = GameMeta.PLAYERS['red']

    cpdef dict get_num_played(self):
        return {'red': self.red_played, 'blue': self.blue_played}

    cpdef dict get_red_groups(self):
        """
        Returns (dict): group of red groups for unionfind check
        """
        return self.red_groups.get_groups()

    cpdef dict get_blue_groups(self):
        """
        Returns (dict): group of red groups for unionfind check
        """
        return self.blue_groups.get_groups()

    cpdef void place_red(self, tuple cell):
        """
        Place a red stone regardless of whose turn it is.
        Args:
            cell (tuple): row and column of the cell
        """
        if self.board[cell[0], cell[1]] == GameMeta.PLAYERS['none']:
            self.board[cell[0], cell[1]] = GameMeta.PLAYERS['red']
            self.red_played += 1
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a red edge connect it appropriately
        if cell[0] == 0:
            self.red_groups.join(GameMeta.EDGE1, cell)
        if cell[0] == self.size - 1:
            self.red_groups.join(GameMeta.EDGE2, cell)
        # join any groups connected by the new red stone
        for n in self.neighbors(cell):
            if self.board[n[0], n[1]] == GameMeta.PLAYERS['red']:
                self.red_groups.join(n, cell)

    cpdef void place_blue(self, tuple cell):
        """
        Place a blue stone regardless of whose turn it is.
        Args:
            cell (tuple): row and column of the cell
        """
        if self.board[cell[0], cell[1]] == GameMeta.PLAYERS['none']:
            self.board[cell[0], cell[1]] = GameMeta.PLAYERS['blue']
            self.blue_played += 1
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a blue edge connect it appropriately
        if cell[1] == 0:
            self.blue_groups.join(GameMeta.EDGE1, cell)
        if cell[1] == self.size - 1:
            self.blue_groups.join(GameMeta.EDGE2, cell)
        # join any groups connected by the new blue stone
        for n in self.neighbors(cell):
            if self.board[n[0], n[1]] == GameMeta.PLAYERS['blue']:
                self.blue_groups.join(n, cell)
             
    cpdef int turn(self):
        """
        Return the player with the next move.
        """
        return self.to_play

    cpdef void set_turn(self, int player):
        """
        Set the player to take the next move.
        Raises:
            ValueError if player turn is not 1 or 2
        """
        if player in GameMeta.PLAYERS.values() and player != GameMeta.PLAYERS['none']:
            self.to_play = player
        else:
            raise ValueError('Invalid turn: ' + str(player))

    cpdef int winner(self):
        """
        Return a number corresponding to the winning player,
        or none if the game is not over.
        """
        if self.red_groups.connected(GameMeta.EDGE1, GameMeta.EDGE2):
            return GameMeta.PLAYERS['red']
        elif self.blue_groups.connected(GameMeta.EDGE1, GameMeta.EDGE2):
            return GameMeta.PLAYERS['blue']
        else:
            return GameMeta.PLAYERS['none']

    cpdef list neighbors(self, tuple cell):
        """
        Return list of neighbors of the passed cell.
        Args:
            cell tuple):
        """
        x = cell[0]
        y = cell[1]
        return [(n[0] + x, n[1] + y) for n in GameMeta.NEIGHBOR_PATTERNS
                if (0 <= n[0] + x < self.size and 0 <= n[1] + y < self.size)]

    cpdef list moves(self):
        """
        Get a list of all moves possible on the current board.
        """
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[x, y] == GameMeta.PLAYERS['none']:
                    moves.append((x, y))
        return moves