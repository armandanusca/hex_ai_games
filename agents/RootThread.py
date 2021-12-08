import multiprocessing as mlp

class RootThread(mlp.Process):
  """
  Implementation of Root parallelization in MCTS agent

  """
  def __init__(self, agent, time, q1, q2):
    mlp.Process.__init__(self)
    self.agent = agent
    self.agent_q = q1
    self.moves_q = q2
    self.time = time

  def run(self):
    self.agent.search(self.time)
    moves = {}
    for child in self.agent.root.children.values():
      moves[child.move] = child.counter_visits
    self.agent_q.put(self.agent)
    self.moves_q.put(moves)

  def get_result(self):
    return self.agent_q.get()

  def get_moves(self):
    return self.moves_q.get()