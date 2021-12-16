# -----------------------------------------------------------
# Group 4 - A Monte Carlo Tree Search based agent for playing
# hex
# -----------------------------------------------------------

import socket

from gamestate import GameState
from RootThreadingAgent import RootThreadingAgent
from utils import extract_last_move_from_board

import argparse

parser = argparse.ArgumentParser(description='Parallelization Agent')
parser.add_argument('--processes', '-p', type=int, default=1, dest = 'processes',
                    help='Number f processes to use')
args = parser.parse_args()

class MCTSAgent():
    """
    A class to represent the agent.
    ...

    Attributes
    ----------
    host : str
        host to connect with other agents
    port : str
        port used for the socket
    time_limit : int
        maximum number of seconds allowed per move
    agent: Agent
        Agent engine object used to compute moves


    Methods
    -------
    run():
        Reads data until it receives an END message or the socket closes.
    interpret_data(data):
        Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
    test_swap(action):
        Decides if it is advantageous to swap
        based on the previous move
    choose_move():
        Invoke the engine behind the agent
        Perform a search for a limited amount of time
        Get the best move and send it
    make_move(action=None):
        Makes a move from the available pool of choices. 
        If it considers that is advantageous to swap it will do so.
        When it does that, it resets the game state in the agent.
    opp_colour():
        Returns the char representation of the colour opposite to the
        current one.
    """

    host = "127.0.0.1"
    port = 1234
    time_limit = 4
    agent = None

    def __init__(self, board_size=11):
        """
        Constructs all the necessary attributes for the Agent object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.host, self.port))

        self.board_size = board_size
        self.colour = ""
        self.turn_count = 0
        self.agent = RootThreadingAgent(GameState(board_size), processes=args.processes)

    def run(self):
        """
        Reads data until it receives an END message or the socket closes.
        """

        while True:
            data = self.s.recv(1024)
            if not data:
                break

            if (self.interpret_data(data)):
                break

    def interpret_data(self, data) -> bool:
        """
        Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]

        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        last_move = extract_last_move_from_board(s[2])
                        self.agent = RootThreadingAgent(GameState(11), processes=args.processes)
                        self.agent.move((last_move[0], last_move[1]))
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.agent.move((action[0], action[1]))
                    self.make_move(action)
        return False

    def test_swap(self, action) -> bool:
        '''
        Decides if it is advantageous to swap
        based on the previous move 

            Parameters:
                    action (tuple): Coordinates of the previous move

            Returns:
                    (bool): If a swap is recommended
        '''
        # TODO: Implement a way to decide if the agent should swap
        if action:
            return True
        return False

    def choose_move(self) -> None:
        """
        Invoke the engine behind the agent
        Perform a search for a limited amount of time
        Get the best move and send it
        """
        self.agent.search(self.time_limit)


        move = self.agent.best_move()
        # print("Best move suggested: ", move)
        self.agent.move(move)

        # Performance measures
        num_rollouts, node_count, run_time = self.agent.statistics()
        print(num_rollouts, node_count, run_time)
        # Send move
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))

    def make_move(self, action=None) -> None:
        '''
        Makes a move from the available pool of choices. 
        If it considers that is advantageous to swap it will do so.
        When it does that, it resets the game state in the agent.

            Parameters:
                    action (tuple): Coordinates of the previous move
        '''
        if self.colour == "B" and self.turn_count == 0:
            if self.test_swap(action):
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                # self.colour = self.opp_colour()
                self.agent = RootThreadingAgent(GameState(11), processes=args.processes)
                self.agent.move((action[0], action[1]))
            else:
                self.choose_move()
        else:
            self.choose_move()
        self.turn_count += 1

    def opp_colour(self):
        """
        Returns the char representation of the colour opposite to the
        current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"

if (__name__ == "__main__"):
    agent = MCTSAgent()
    agent.run()
