from sys import argv, float_repr_style, platform
from os.path import realpath, sep
from Game import Game

def main():
    player1 = {
        "name": "rave1",
        "run string": "python3 ../agents/mcts_agent.py"
    }
    player2 = {
        "name": "rave2",
        "run string": "python3 ../agents/mcts_agent.py"
    }

    g = Game(
        board_size=11,
        player1=player1, player2=player2,
        verbose=False,
        log=True,
        print_protocol=False,
        kill_bots=False,
        silent_bots=False
    )
    g.run()

if __name__ == "__main__":
    main()