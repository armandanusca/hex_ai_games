import copy
import multiprocessing as mlp
import random

from meta import GameMeta


class LeafThread(mlp.Process):
    """
    This class uses parallel threads for parallelizing playout phase

    """

    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.moves = state.moves()
        self.blue_rave_pts = []
        self.red_rave_pts = []
        mlp.Process.__init__(self)

    def run(self):
        moves = self.moves
        while self.state.winner == GameMeta.PLAYERS["none"]:
            move = random.choice(moves)
            self.state.play(move)
            moves.remove(move)

        for x in range(self.state.size):
            for y in range(self.state.size):
                if self.state.board[(x, y)] == GameMeta.PLAYERS["blue"]:
                    self.blue_rave_pts.append((x, y))
                elif self.state.board[(x, y)] == GameMeta.PLAYERS["red"]:
                    self.red_rave_pts.append((x, y))

    def get_results(self):
        return self.state.winner, self.blue_rave_pts, self.red_rave_pts
