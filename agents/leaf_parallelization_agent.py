import random
import threading
from copy import deepcopy
from time import time

from gamestate import GameState
from meta import GameMeta
from rave_mcts import RaveMCTSEngine


class LeafThread(threading.Thread):
    """
    This class uses parallel threads for parallelizing playout phase 
    """

    def __init__(self, state):
        self.state = deepcopy(state)
        self.moves = state.moves()
        self.blue_rave_pts = []
        self.red_rave_pts = []
        threading.Thread.__init__(self)

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


class LeafThreadingAgent(RaveMCTSEngine):
    """
    Basic no frills implementation of an agent that preforms MCTS for hex.
    """
    EXPLORATION = 1

    def __init__(self, state=GameState(11), threads=10):
        super().__init__(state)
        self.threads = threads

    def search(self, time_budget):
        """
        Search and update the search tree for a 
        specified amount of time in secounds.
        """
        start_time = time()
        num_rollouts = 0

        # do until we exceed our time budget
        while (time() - start_time < time_budget):
            node, state = self.select_node()
            turn = state.turn()
            threads = []
            for i in range(self.threads):
                threads.append(LeafThread(state))
                threads[i].start()
            for t in threads:
                t.join()

            total_outcome = []
            total_blue_rave_pts = []
            total_red_rave_pts = []

            for t in threads:
                outcome, blue_rave_pts, red_rave_pts = t.get_results()
                # total_outcome.append(outcome)
                # total_blue_rave_pts.extend(blue_rave_pts)
                # total_red_rave_pts.extend(red_rave_pts)
                self.backup(node, turn, outcome, blue_rave_pts, red_rave_pts)
                num_rollouts += 1

            # self.parallel_backup(node, turn, total_outcome, total_blue_rave_pts, total_red_rave_pts)
        run_time = time() - start_time
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
