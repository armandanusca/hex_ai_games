import multiprocessing as mlp
import copy
import random

class LeafThread(mlp.Process):
  """
  This class uses parallel threads for parallelizing playout phase 

  """
  def __init__(self, state):
    self.state = copy.deepcopy(state)
    self.moves = state.moves()
    mlp.Process.__init__(self)

  def run(self):
    while (self.state.winner == 0):
      move = random.choice(self.moves)
      self.state.play(move)
      self.moves.remove(move)

  def get_results(self):
    return self.state.winner