from mcts_agent import *
from LeafThread import LeafThread
from time import perf_counter


class LeafThreadingAgent(RaveMCTSEngine):
  """
  Basic no frills implementation of an agent that preforms MCTS for hex.

  """
  EXPLORATION = 1

  def __init__(self, state=GameState(11), processes=2):
    super().__init__(state)
    self.processes = processes

  def search(self, time_budget):
    """
    Search and update the search tree for a 
    specified amount of time in secounds.

    """
    startTime = perf_counter()
    num_rollouts = 0

    # do until we exceed our time budget
    while (perf_counter() - startTime < time_budget):
      node, state = self.select_node()
      turn = state.turn()
      processes = []
      for i in range(self.processes):
        processes.append(LeafThread(state))
        processes[i].start()
      for t in processes:
        t.join()

      outcome = [t.get_results() for t in processes]
      self.backup(node, turn, outcome)
      num_rollouts += self.processes
    run_time = perf_counter() - startTime
    node_count = self.tree_size()
    self.run_time = run_time
    self.node_count = node_count
    self.num_rollouts = num_rollouts

  def backup(self, node, turn, outcome):
    """
    Update the node statistics on the path from the passed node to root to reflect
    the outcome of a randomly simulated playout.

    """
    # Careful: The reward is calculated for player who just played
    # at the node and not the next player to play
    score = 0
    for res in outcome:
      if res == turn:
        score -=1
      else:
        score +=1
    num = len(outcome)
    reward = score

    while node != None:
      node.counter_visits += num
      node.reward_average += reward
      node = node.parent
      reward = -reward 

