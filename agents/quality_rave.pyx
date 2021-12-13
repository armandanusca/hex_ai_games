# keep this line for cython directives

from copy import deepcopy
from libc.math cimport sqrt, log
from libc.stdlib cimport rand
from queue import Queue
from time import time
from numpy import where
import numpy as np
cimport numpy as np
import cython

from gamestate cimport GameState
from meta import GameMeta, MCTSMeta
from operator import itemgetter

np.import_array()
DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef extern from "<math.h>" nogil:
    float fmaxf(float, float)
    double exp(double)
 
cdef class RollingStatistic():

    cdef public:
        int n
        double mean
        double M2
        double delta

    def __init__(self):
        self.n, self.mean, self.M2, self.delta = 0, 0.0, 0.0, 0.0

    cdef void update(self, float new_value):
        self.n += 1
        self.delta = new_value - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (new_value - self.mean)

    cdef double variance(self):
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n)

    cdef double std(self):
        return sqrt(self.variance())

    cdef void clear(self):
        self.n, self.mean, self.M2 = 0, 0.0, 0.0

@cython.wraparound(False)
@cython.boundscheck(False)
cdef cchoice(arr):       
    return arr[rand() % len(arr)] 

@cython.wraparound(False)
@cython.boundscheck(False)
cdef bint expand(Node parent, GameState state):
    """
    Generate the children of the passed "parent" node based on the available
    moves in the passed gamestate and add them to the tree.

    Returns:
        object:
    """

    if state.winner() != GameMeta.PLAYERS["none"]:
        # game is over at this node so nothing to expand
        return False

    #for move in state.moves():
    #    children.append(Node(move, parent))
    children = [Node(move, parent) for move in state.moves()]

    parent.add_children(children)
    return True

cdef tuple roll_out(state):
        """
        Simulate a random game except that we play all known critical
        cells first, return the winning player and record critical cells at the end.

        """

        cdef:
            (int, int) move
            np.ndarray[DTYPE_t, ndim=1] blue_rave_ptsx, blue_rave_ptsy, red_rave_ptsx, red_rave_ptsy

        moves = state.moves()
        while state.winner() == GameMeta.PLAYERS["none"]:
            move = cchoice(moves)
            state.play(move)
            moves.remove(move)

        players_moves = state.get_rb_played()

        blue_rave_ptsx, blue_rave_ptsy = where(state.board == GameMeta.PLAYERS["blue"])
        red_rave_ptsx, red_rave_ptsy = where(state.board == GameMeta.PLAYERS["red"])

        return state.winner(), players_moves, red_rave_ptsx, red_rave_ptsy, blue_rave_ptsx, blue_rave_ptsy
        
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

    cpdef float value(self, float explore = MCTSMeta.EXPLORATION, float rave_const = MCTSMeta.RAVE_CONST):
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
            alpha = fmaxf(0, (rave_const - self.counter_visits) / rave_const)
            UCT = self.reward_average / self.counter_visits + explore * sqrt(
                2 * log(self.parent.counter_visits) / self.counter_visits)
            AMAF = self.rave_reward_average / self.rave_counter_visits if self.rave_counter_visits != 0 else 0
            v = (1 - alpha) * UCT + alpha * AMAF
            #if v != v:
            #    print("nan")
            #    print(alpha, UCT, AMAF)
            #    print(self.rave_reward_average)
            return (1 - alpha) * UCT + alpha * AMAF


cdef class QRAVEEngine():

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

        float a_const
        float k_const

        RollingStatistic rs1, rs2


    def __init__(self, state: GameState = GameState(11)):
        self.root_state = deepcopy(state)
        self.root = Node()
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

        self.a_const = MCTSMeta.A_CONST
        self.k_const = MCTSMeta.K_CONST
        self.rs1 = RollingStatistic()
        self.rs2 = RollingStatistic()

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
        bestchild = cchoice(max_nodes)
        return bestchild.move

    cpdef void search(self, int time_budget):
        """
        Search and update the search tree for a specified amount of time in seconds.
        """

        cdef:
            long start_time
            int num_rollouts, turn
            Node node
            GameState state
            int outcome
            np.ndarray[DTYPE_t, ndim=1] red_rave_ptsx, red_rave_ptsy, blue_rave_ptsx, blue_rave_ptsy

        start_time = time()
        num_rollouts = 0

        # do until we exceed our time budget
        while time() - start_time < time_budget:
            node, state = self.select_node()
            turn = state.turn()
            outcome, players_moves, red_rave_ptsx, red_rave_ptsy, blue_rave_ptsx, blue_rave_ptsy = roll_out(state)
            self.backprop(node, turn, outcome, players_moves, red_rave_ptsx, red_rave_ptsy, blue_rave_ptsx, blue_rave_ptsy)
            num_rollouts += 1

        run_time = time() - start_time
        node_count = self.tree_size()
        self.run_time = run_time
        self.node_count = node_count
        self.num_rollouts = num_rollouts

    cdef tuple select_node(self):
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
        while node.children:
            n_values = [(n.value(), n) for n in node.children.values()]
            max_value = max(n_values, key=itemgetter(0))[0]
            n_values = [n[1] for n in n_values if n[0] == max_value]

            node = cchoice(n_values)
            state.play(node.move)

            # if some child node has not been explored select it before expanding
            # other children
            if node.counter_visits == 0:
                return node, state

        # if we reach a leaf node generate its children and return one of them
        # if the node is terminal, just return the terminal node
        if expand(node, state):
            node = cchoice(list(node.children.values()))
            state.play(node.move)
        return node, state

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)
    cdef void backprop(self, Node node, int turn, int outcome, tuple players_moves, np.ndarray[DTYPE_t, ndim=1] red_rave_ptsx, np.ndarray[DTYPE_t, ndim=1] red_rave_ptsy, np.ndarray[DTYPE_t, ndim=1] blue_rave_ptsx, np.ndarray[DTYPE_t, ndim=1] blue_rave_ptsy):
        """
        Update the node statistics on the path from the passed node to root to reflect
        the outcome of a randomly simulated playout.
        """
        # note that reward is calculated for player who just played
        # at the node and not the next player to play
        cdef:
            int index
            double temp_reward
            (int, int) point

        reward = -1 if outcome == turn else 1

        self.rs1.update(players_moves[0])
        self.rs2.update(players_moves[1])

        qb = self.compute_reward(players_moves)

        if self.num_rollouts == 0:
            self.rs1.clear()
            self.rs2.clear()

        while node != None:
            if turn == GameMeta.PLAYERS["red"]:
                temp_reward = reward + (reward * self.a_const * qb[0])
                node.rave_reward_average += temp_reward
                for index in range(red_rave_ptsx.shape[0]):
                    point = (red_rave_ptsx[index], red_rave_ptsy[index])
                    if point in node.children:
                        node.children[point].rave_reward_average += -temp_reward
                        node.children[point].rave_counter_visits += 1
            else:
                temp_reward = reward + (reward * self.a_const * qb[1])
                node.rave_reward_average += temp_reward
                for index in range(blue_rave_ptsx.shape[0]):
                    point = (blue_rave_ptsx[index], blue_rave_ptsy[index])
                    if point in node.children:
                        node.children[point].rave_reward_average += -temp_reward
                        node.children[point].rave_counter_visits += 1

            node.counter_visits += 1
            node.reward_average += reward
            turn = GameMeta.PLAYERS['red'] if turn == GameMeta.PLAYERS['blue'] else GameMeta.PLAYERS['blue']
            reward = -reward
            node = node.parent

    cpdef tuple statistics(self):
        return self.num_rollouts, self.node_count, self.run_time

    cdef tuple compute_reward(self, tuple player_length):
        cdef:
            double mean1, mean2, deviation1, deviation2, mean_offset1, mean_offset2, lmdb1, lmdb2, bonus1, bonus2

        mean1 = self.rs1.mean
        mean2 = self.rs2.mean

        deviation1 = self.rs1.std()
        deviation2 = self.rs2.std()

        mean_offset1 = mean1 - player_length[0]
        mean_offset2 = mean2 - player_length[1]

        lmdb1 = mean_offset1 / deviation1 if deviation1 != 0 else 0
        lmdb2 = mean_offset2 / deviation2 if deviation2 != 0 else 0

        bonus1 = -1 + (2 / (1 + exp(-lmdb1 * self.k_const)))
        bonus2 = -1 + (2 / (1 + exp(-lmdb2 * self.k_const)))          

        return (bonus1, bonus2)

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
