from time import perf_counter

from gamestate import GameState
from LeafThread import LeafThread
from mcts_agent import RaveMCTSEngine
from meta import GameMeta


class LeafThreadingAgent(RaveMCTSEngine):
    """
    Basic no frills implementation of an agent that preforms MCTS for hex.

    """
    EXPLORATION = 1

    def __init__(self, state=GameState(11), processes=6):
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

            # outcome = [t.get_results() for t in processes]
            # self.backup(node, turn, outcome)
            # total_outcome = []
            # total_blue_rave_pts = []
            # total_red_rave_pts = []

            for t in processes:
                outcome, blue_rave_pts, red_rave_pts = t.get_results()
                # total_outcome.append(outcome)
                # total_blue_rave_pts.extend(blue_rave_pts)
                # total_red_rave_pts.extend(red_rave_pts)
                self.backup(node, turn, outcome, blue_rave_pts, red_rave_pts)
                # num_rollouts += 1

            # self.parallel_backup(node, turn, total_outcome, total_blue_rave_pts, total_red_rave_pts)
            num_rollouts += self.processes
        run_time = perf_counter() - startTime
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = num_rollouts

    def backup(self, node, turn, outcome, blue_rave_pts, red_rave_pts):
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play
        reward = -1 if outcome == turn else 1

        while node is not None:
            if turn == GameMeta.PLAYERS["red"]:
                for point in red_rave_pts:
                    if point in node.children:
                        node.children[point].rave_reward_average += -reward
                        node.children[point].rave_counter_visits += 1
            else:
                for point in blue_rave_pts:
                    if point in node.children:
                        node.children[point].rave_reward_average += -reward
                        node.children[point].rave_counter_visits += 1

            node.counter_visits += 1
            node.reward_average += reward
            turn = GameMeta.PLAYERS['red'] if turn == GameMeta.PLAYERS['blue'] else GameMeta.PLAYERS['blue']
            reward = -reward
            node = node.parent

    def parallel_backup(self, node, turn, outcome, blue_rave_pts, red_rave_pts):
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play
        score = 0
        for res in outcome:
            if res == turn:
                score -= 1
            else:
                score += 1
        num = len(outcome)
        reward = score

        while node is not None:
            if turn == GameMeta.PLAYERS["red"]:
                for point in red_rave_pts:
                    if point in node.children:
                        node.children[point].rave_reward_average += -reward
                        node.children[point].rave_counter_visits += num
            else:
                for point in blue_rave_pts:
                    if point in node.children:
                        node.children[point].rave_reward_average += -reward
                        node.children[point].rave_counter_visits += num

            node.counter_visits += num
            node.reward_average += reward
            turn = GameMeta.PLAYERS['red'] if turn == GameMeta.PLAYERS['blue'] else GameMeta.PLAYERS['blue']
            reward = -reward
            node = node.parent
