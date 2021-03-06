# keep this line for cython directives

from copy import deepcopy
from libc.math cimport sqrt, log
from queue import Queue
from random import choice
from time import time
from numpy import where

from gamestate cimport GameState
from meta import GameMeta, MCTSMeta
from operator import itemgetter

cdef extern from "<math.h>" nogil:
    float fmaxf(float, float)

cdef class Node:
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
    rave_counter_visits: int
        times this move has appeared in a rollout
    rave_reward_average: int
        times this move has been critical in a rollout (lead to an outcome)

    Methods
    -------
    add_children(children: dict):
        Add a list of nodes to the children of this node.
    """

    cdef public:
        tuple move
        Node parent
        dict children
        int outcome
        int counter_visits
        float reward_average
        int rave_counter_visits
        float rave_reward_average

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

    cpdef void add_children(self, list children):
        """
        Add a list of nodes to the children of this node.
        """
        for child in children:
            self.children[child.move] = child

    cpdef float value(self, float explore = MCTSMeta.EXPLORATION, double rave_const = 0.00000016):
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
        cdef float alpha, UCT, AMAF

        # unless explore is set to zero, maximally favor unexplored nodes
        if self.counter_visits == 0:
            return 0 if explore == 0 else GameMeta.INF
        else:
            # rave valuation:
            #alpha = fmaxf(0, (rave_const - self.counter_visits) / rave_const)
            alpha = self.rave_counter_visits / (self.rave_counter_visits + self.counter_visits + 4 * self.rave_counter_visits * self.counter_visits * rave_const)
            UCT = self.reward_average / self.counter_visits + explore * sqrt(
                2 * log(self.parent.counter_visits) / self.counter_visits)
            AMAF = self.rave_reward_average / self.rave_counter_visits if self.rave_counter_visits != 0 else 0
            return (1 - alpha) * UCT + alpha * AMAF


cdef class RaveMCTSEngine():

    """
    Implementation of an agent that performs MCTS with RAVE. It is used for Monte Carlo Tree Search.
    RAVE stands for Rapid Action Value Estimation. It is an optimization strategy for the learning 
    occurred inside the game tree. It contains latest move applied from parent to current node,
    performance metrics, parent node, children nodes and outcome.
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

    cdef public:
        GameState root_state
        Node root
        int node_count
        int run_time
        int num_rollouts

    def __init__(self, state: GameState = GameState(11)):
        self.root_state = deepcopy(state)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    cpdef void set_gamestate(self, object state):
        """
        Set the root_state of the tree to the passed gamestate, this clears all
        the information stored in the tree since none of it applies to the new
        state.
        """
        self.root_state = deepcopy(state)
        self.root = Node()

    cpdef void move(self, tuple move):
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

    cpdef best_move(self):
        """
        Return the best move according to the current tree.
        Returns:
            best move in terms of the most simulations number unless the game is over
        """

        cdef:
            list max_nodes
            int max_value
            Node bestchild

        if self.root_state.winner() != GameMeta.PLAYERS['none']:
            return GameMeta.GAME_OVER

        # choose the move of the most simulated node breaking ties randomly
        max_nodes = [(n.counter_visits, n) for n in self.root.children.values()]
        max_value = max(max_nodes, key=itemgetter(0))[0]
        max_nodes = [n[1] for n in max_nodes if n[0] == max_value]
        #max_value = max(self.root.children.values(), key=lambda n: n.counter_visits).counter_visits
        #max_nodes = [n for n in self.root.children.values() if n.counter_visits == max_value]
        bestchild = choice(max_nodes)
        return bestchild.move

    cpdef void search(self, int time_budget):
        """
        Search and update the search tree for a specified amount of time in seconds.
        """

        start_time = time()
        num_rollouts = 0

        # do until we exceed our time budget
        while time() - start_time < time_budget:
            node, state = self.select_node()
            turn = state.turn()
            outcome, blue_rave_pts, red_rave_pts = RaveMCTSEngine.roll_out(state)
            self.backup(node, turn, outcome, blue_rave_pts, red_rave_pts)
            num_rollouts += 1
        run_time = time() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = num_rollouts

    cpdef select_node(self):
        """
        Select a node in the tree to preform a single simulation from.
        """

        cdef:
            Node node
            GameState state
            list n_values
            float max_value

        node = self.root
        state = deepcopy(self.root_state)

        # stop if we reach a leaf node
        while len(node.children) != 0:
        # ARMAND: Please check code to see if it is correct
            n_values = [(n.value(), n) for n in node.children.values()]
            max_value = max(n_values, key=itemgetter(0))[0]
            n_values = [n[1] for n in n_values if n[0] == max_value]
            # max_value = max(node.children.values(),
                           #  key=lambda n:
                          #   n.value()).value()
            # descend to the maximum value node, break ties at random
        #   max_nodes = [n for n in node.children.values() if
             #             n.value() == max_value]
            node = choice(n_values)
            state.play(node.move)

            # if some child node has not been explored select it before expanding
            # other children
            if node.counter_visits == 0:
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if RaveMCTSEngine.expand(node, state):
            node = choice(list(node.children.values()))
            state.play(node.move)
        return node, state

    @staticmethod
    def expand(parent, state):
        """
        Generate the children of the passed "parent" node based on the available
        moves in the passed gamestate and add them to the tree.

        Returns:
            object:
        """
        children = []
        if state.winner() != GameMeta.PLAYERS["none"]:
            # game is over at this node so nothing to expand
            return False

        for move in state.moves():
            children.append(Node(move, parent))

        parent.add_children(children)
        return True

    @staticmethod
    def roll_out(state):
        """
        Simulate a random game except that we play all known critical
        cells first, return the winning player and record critical cells at the end.

        """
        moves = state.moves()
        while state.winner() == GameMeta.PLAYERS["none"]:
            move = choice(moves)
            state.play(move)
            moves.remove(move)

        blue_rave_pts = [(x,y) for x,y in zip(*where(state.board == GameMeta.PLAYERS["blue"]))]
        red_rave_pts = [(x,y) for x,y in zip(*where(state.board == GameMeta.PLAYERS["red"]))]

        #for x in range(state.size):
        #    for y in range(state.size):
        #        if state.board[(x, y)] == GameMeta.PLAYERS["blue"]:
        #            blue_rave_pts.append((x, y))
        #        elif state.board[(x, y)] == GameMeta.PLAYERS["red"]:
        #            red_rave_pts.append((x, y))

        return state.winner(), blue_rave_pts, red_rave_pts

    cpdef void backup(self, Node node, int turn, int outcome, list blue_rave_pts, list red_rave_pts):
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

    cpdef tuple statistics(self):
        return self.num_rollouts, self.node_count, self.run_time

    cpdef int tree_size(self):
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
