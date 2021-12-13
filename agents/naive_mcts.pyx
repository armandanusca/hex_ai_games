# keep this line for cython directives

from copy import deepcopy
from math import log, sqrt
from queue import Queue
from random import choice
from time import time

from gamestate import GameState
from meta import GameMeta, MCTSMeta


class Node:
    """
    A class to represent a node from the Game Tree. It is used for Monte Carlo Tree Search.
    It contains latest move applied from parent to current node, performance metrics,
    parent node, children nodes and outcome.
    ...

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

    def add_children(self, children: list) -> None:
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[child.move] = child

    @property
    def value(self, explore: float = MCTSMeta.EXPLORATION):
        '''
        Calculate the Upper Confidence bounds applied to Trees(UCT) value 
        of this node relative to its parent

            Parameters:
                    explore (float): how much the value should favor nodes 
                                    that have yet to be thoroughly explored 
                                    versus nodes that seem to have a high win rate

            Returns:
                    (float): UCT value
        '''

        if self.counter_visits == 0:
            return 0 if explore == 0 else GameMeta.INF
        else:
            # exploitation + exploration formula
            return self.reward_average / self.counter_visits + explore * sqrt(
                2 * log(self.parent.counter_visits) / self.counter_visits)


class NaiveMCTSEngine:
    """
    Implementation of an agent that performs MCTS. It is used for Monte Carlo Tree Search.
    It contains latest move applied from parent to current node, performance metrics,
    parent node, children nodes and outcome.
    ...

    Attributes
    ----------
    root_state : GameState
        object to store the current game situation
    root : Node
        root of the tree search
    node_count : int
        the number of nodes in a tree
    run_time: int
        time taken for each run
    num_rollouts: int
        the number of rollouts for each search
    exploration: int
        specifies how much the value should favor nodes 
        that have yet to be thoroughly explored versus nodes
        that seem to have a high win rate

    Methods
    -------
    search(time_budget: int):
        Search and update the search tree for a
        specified amount of time in seconds.
    select_node():
        Select a node in the tree to preform a single simulation from.
    expand(parent: Node, state: GameState):
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.
    roll_out(state: GameState):
        Simulate an entirely random game from the passed state and return the winning
        player.
    backup(node: Node, turn: int, outcome: int):
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
    best_move():
        Return the best move according to the current tree.
    move(move: tuple):
        Make the passed move and update the tree appropriately.
    set_gamestate(state: GameState):
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state.
    statistics():
        Getter for performance metrics
    tree_size():
        Count nodes in tree by BFS.
    """

    def __init__(self, state=GameState(11)):
        """
        Initialize a new node with optional move and parent and initially empty
        children list and rollout statistics and unspecified outcome.

        Parameters:
                move (tuple): the move that generated the current node
                parent (Node): parent node
        """
        self.root_state = deepcopy(state)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def search(self, time_budget: int) -> None:
        """
        Search and update the search tree for a
        specified amount of time in seconds.
        """
        start_time = time()
        num_rollouts = 0

        # do until we exceed our time budget
        while time() - start_time < time_budget:
            node, state = self.select_node()
            turn = state.turn()
            outcome = self.roll_out(state)
            self.backup(node, turn, outcome)
            num_rollouts += 1
        run_time = time() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.counter_visitsode_count = node_count
        self.counter_visitsum_rollouts = num_rollouts

    def select_node(self) -> tuple:
        """
        Select a node in the tree to preform a single simulation from.
        """
        node = self.root
        state = deepcopy(self.root_state)

        # stop if we find reach a leaf node
        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value).value
            max_nodes = [n for n in node.children.values()
                         if n.value == max_value]
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
            bool: returns false If node is leaf (the game has ended).
        """
        children = []
        if state.winner() != GameMeta.PLAYERS['none']:
            # game is over at this node so nothing to expand
            return False

        for move in state.moves():
            children.append(Node(move, parent))

        parent.add_children(children)
        return True

    @staticmethod
    def roll_out(state: GameState) -> int:
        """
        Simulate an entirely random game from the passed state and return the winning
        player.

        Args:
            state: game state

        Returns:
            int: winner of the game
        """
        moves = state.moves()  # Get a list of all possible moves in current state of the game

        while state.winner() == GameMeta.PLAYERS['none']:
            move = choice(moves)
            state.play(move)
            moves.remove(move)

        return state.winner()

    @staticmethod
    def backup(node: Node, turn: int, outcome: int) -> None:
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.

        Args:
            node:
            turn: winner turn
            outcome: outcome of the rollout

        Returns:
            object:
        """
        # Careful: The reward is calculated for player who just played
        # at the node and not the next player to play
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.counter_visits += 1
            node.reward_average += reward
            node = node.parent
            reward = 0 if reward == 1 else 1

    def best_move(self) -> tuple:
        """
        Return the best move according to the current tree.
        Returns:
            best move in terms of the most simulations number unless the game is over
        """
        if self.root_state.winner() != GameMeta.PLAYERS['none']:
            return GameMeta.GAME_OVER

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.counter_visits).counter_visits
        max_nodes = [n for n in self.root.children.values() if n.counter_visits == max_value]
        bestchild = choice(max_nodes)
        return bestchild.move

    def move(self, move: tuple) -> tuple:
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
            return move

        # if for whatever reason the move is not in the children of
        # the root just throw out the tree and start over
        self.root_state.play(move)
        self.root = Node()
        return move

    def set_gamestate(self, state: GameState) -> None:
        """
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state.

        """
        self.root_state = deepcopy(state)
        self.root = Node()

    def statistics(self) -> tuple:
        return self.counter_visitsum_rollouts, self.counter_visitsode_count, self.run_time

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
