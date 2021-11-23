import socket
from random import choice
from time import sleep
from gamestate import GameState
from lib_mcst import UctMctsAgent


class MCTSAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234
    time_limit = 2
    game_state = GameState(11)
    agent = UctMctsAgent(game_state)

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.board = []
        self.colour = ""
        self.turn_count = 0

    def run(self):
        """Reads data until it receives an END message or the socket closes."""

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if (self.interpret_data(data)):
                break

        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        """Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [
                    [0]*self.board_size for i in range(self.board_size)]

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
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()
                    self.agent.move((action[0],action[1]))
                    self.make_move(action)
        return False

    def test_swap(self, action):
        # TODO: Implement a way to decide if the agent should swap
        if action:
            return True
        return False

    def choose_move_test(self):
        self.agent.search(self.time_limit)
        num_rollouts, node_count, run_time = self.agent.statistics()
        print(num_rollouts, node_count, run_time)
        move = self.agent.best_move()  # the move is tuple like (3, 1)
        print("Best move suggested: ", move)
        self.agent.move(move)
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))
        self.board[move[0]][move[1]] = self.colour

    def make_move(self, action=None):
        """Makes a random move from the available pool of choices. If it can
        swap, chooses to do so 50% of the time.
        """            
        if self.colour == "B" and self.turn_count == 0:
            if self.test_swap(action):
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.colour = self.opp_colour()
                self.game_state = GameState(11)
                self.game_state.play((action[0],action[1]))
            else:
                self.choose_move_test()
        else:
            self.choose_move_test()
        self.turn_count += 1

    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
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
