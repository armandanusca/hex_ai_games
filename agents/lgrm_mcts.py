from copy import deepcopy
from math import log, sqrt
from queue import Queue
from random import choice, random
from time import time

from gamestate import GameState
from meta import GameMeta, MCTSMeta


class Node:
    """
    A class to represent a node from the Game Tree. It is used for Monte Carlo Tree Search.
    It contains latest move applied from parent to current node, performance metrics,
    parent node, children nodes and outcome.


    Attributes
    ----------
    move : tuple
        move which lead from parent to current node
    parent : Node
        parent node
    children : dict
        dictionary of all possible moves from the current node
    outcome: int
        if node is a leaf, then outcome is equal to numeric representation
        of the winner. None otherwise
    counter_visits: int
        times this position was visited
    reward_average: int
        average reward (wins-losses) from this position
    rave_counter_visits: int
        times this move has appeared in a rollout
    rave_reward_average: int
        times this move has been critical in a rollout (lead to an outcome)

    Methods
    -------
    add_children(children: dict):
        Add a list of nodes to the children of this node.
    """

    def __init__(self, move: tuple = None, parent: object = None):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.

        Parameters:
                move (tuple): the move that generated the current node
                parent (Node): parent node
        """
        self.move = move
        self.parent = parent
        self.children = {}
        self.outcome = GameMeta.PLAYERS['none']

        # performance metrics
        self.counter_visits = 0  # times this position was visited
        self.reward_average = 0  # average reward (wins-losses) from this position
        self.rave_counter_visits = 0  # times this move has appeared in a rollout
        self.rave_reward_average = 0  # times this move has been critical in a rollout

    def add_children(self, children: dict) -> None:
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[child.move] = child

    @property
    def value(self, explore: float = MCTSMeta.EXPLORATION, rave_const: float = MCTSMeta.RAVE_CONST) -> float:
        '''
        Calculate the evaluation formula applied to the Game Tree

            Parameters:
                    explore (float): how much the value should favor nodes 
                                    that have yet to be thoroughly explored 
                                    versus nodes that seem to have a high win rate

                    rave_const (float): constant to quantify how to balance between UCT and AMAF

            Returns:
                    (float): node score
        '''

        # unless explore is set to zero, maximally favor unexplored nodes
        if self.counter_visits == 0:
            return 0 if explore == 0 else GameMeta.INF
        else:
            # rave valuation:
            alpha = max(0, (rave_const - self.counter_visits) / rave_const)
            UCT = self.reward_average / self.counter_visits + explore * sqrt(
                2 * log(self.parent.counter_visits) / self.counter_visits)
            AMAF = self.rave_reward_average / self.rave_counter_visits if self.rave_counter_visits != 0 else 0
            return (1 - alpha) * UCT + alpha * AMAF


class LGRMCTSEngine():

    def __init__(self, state: GameState = GameState(11)):
        self.root_state = deepcopy(state)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.blue_reply = {}
        self.red_reply = {}

    def set_gamestate(self, state: GameState) -> None:
        """
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state. It only saves replies from previous simulations 
        to use in the Last Good Reply policy.
        """
        self.root_state = deepcopy(state)
        self.root = Node()
        self.red_reply = {}
        self.blue_reply = {}

    def roll_out(self, state: GameState) -> tuple:
        """
        Simulate a random game except that we play all known critical
        cells first, return the winning player and record critical cells at the end.

        """
        moves = state.moves()
        first = state.turn()
        if first == GameMeta.PLAYERS["blue"]:
            current_reply = self.blue_reply
            other_reply = self.red_reply
        else:
            current_reply = self.red_reply
            other_reply = self.blue_reply
        blue_moves = []
        red_moves = []
        last_move = None
        while state.winner == GameMeta.PLAYERS["none"]:
            if last_move in current_reply:
                move = current_reply[last_move]
                if move not in moves or random() > MCTSMeta.RANDOMNESS:
                    move = choice(moves)
            else:
                move = choice(moves)
            if state.turn() == GameMeta.PLAYERS["blue"]:
                blue_moves.append(move)
            else:
                red_moves.append(move)
            current_reply, other_reply = other_reply, current_reply
            state.play(move)
            moves.remove(move)
            last_move = move

        blue_rave_pts = []
        red_rave_pts = []

        for x in range(state.size):
            for y in range(state.size):
                if state.board[(x, y)] == GameMeta.PLAYERS["blue"]:
                    blue_rave_pts.append((x, y))
                elif state.board[(x, y)] == GameMeta.PLAYERS["red"]:
                    red_rave_pts.append((x, y))

        # This part of the algorithm probably deals with adjusting
        # the indices of the arrays.

        offset = 0
        skip = 0
        if state.winner == GameMeta.PLAYERS["blue"]:

            if first == GameMeta.PLAYERS["blue"]:
                offset = 1
            if state.turn() == GameMeta.PLAYERS["blue"]:
                skip = 1
            for i in range(len(red_moves) - skip):
                self.blue_reply[red_moves[i]] = blue_moves[i + offset]
        else:
            if first == GameMeta.PLAYERS["red"]:
                offset = 1
            if state.turn() == GameMeta.PLAYERS["red"]:
                skip = 1
            for i in range(len(blue_moves) - skip):
                self.red_reply[blue_moves[i]] = red_moves[i + offset]

        return state.winner, blue_rave_pts, red_rave_pts

    def move(self, move: tuple) -> None:
        """
        Make the passed move and update the tree appropriately. It is
        designed to let the player choose an action manually (which might
        not be the best action).
        Args:
            move:
        """
        if move in self.root.children:
            child = self.root.children[move]
            child.parent = None
            self.root = child
            self.root_state.play(child.move)
            return

        # if for whatever reason the move is not in the children of
        # the root just throw out the tree and start over
        self.root_state.play(move)
        self.root = Node()

    def best_move(self) -> tuple:
        """
        Return the best move according to the current tree.
        Returns:
            best move in terms of the most simulations number unless the game is over
        """
        if self.root_state.winner != GameMeta.PLAYERS['none']:
            return GameMeta.GAME_OVER

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.counter_visits).counter_visits
        max_nodes = [n for n in self.root.children.values() if n.counter_visits == max_value]
        bestchild = choice(max_nodes)
        return bestchild.move

    def search(self, time_budget: int) -> None:
        """
        Search and update the search tree for a specified amount of time in seconds.
        """
        start_time = time()
        num_rollouts = 0

        # do until we exceed our time budget
        while time() - start_time < time_budget:
            node, state = self.select_node()
            turn = state.turn()
            outcome, blue_rave_pts, red_rave_pts = self.roll_out(state)
            self.backup(node, turn, outcome, blue_rave_pts, red_rave_pts)
            num_rollouts += 1
        run_time = time() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = num_rollouts

    def select_node(self) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = deepcopy(self.root_state)

        # stop if we reach a leaf node
        while len(node.children) != 0:
            max_value = max(node.children.values(),
                            key=lambda n:
                            n.value).value
            # descend to the maximum value node, break ties at random
            max_nodes = [n for n in node.children.values() if
                         n.value == max_value]
            node = choice(max_nodes)
            state.play(node.move)

            # if some child node has not been explored select it before expanding
            # other children
            if node.counter_visits == 0:
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if self.expand(node, state):
            node = choice(list(node.children.values()))
            state.play(node.move)
        return node, state

    @staticmethod
    def expand(parent: Node, state: GameState) -> bool:
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.

        Returns:
            object:
        """
        children = []
        if state.winner != GameMeta.PLAYERS["none"]:
            # game is over at this node so nothing to expand
            return False

        for move in state.moves():
            children.append(Node(move, parent))

        parent.add_children(children)
        return True

    def backup(self, node: Node, turn: int, outcome: int, blue_rave_pts: list, red_rave_pts: list) -> None:
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

    def statistics(self) -> tuple:
        return self.num_rollouts, self.node_count, self.run_time

    def tree_size(self) -> int:
        """
        Count nodes in tree by BFS.
        """
        Q = Queue()
        count = 0
        Q.put(self.root)
        while not Q.empty():
            node = Q.get()
            count += 1
            for child in node.children.values():
                Q.put(child)
        return count
