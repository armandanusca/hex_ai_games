from sys import argv, float_repr_style, platform
from os.path import realpath, sep
from Game import Game
import time

def main():
    for i in range(1,31):
        for j in range(1,31):
            play(i, j)

def play(i, j):
    rave_const_1 = 10 * i
    rave_const_2 = 10 * j
    explore_const= 0.7

    player1 = {
        "name": "rave1",
        "run string": "python3 ../agents/mcts_agent.py r={rave_const} e={explore_const}".format(rave_const=rave_const_1, explore_const=explore_const),
        "hyperparameters": [explore_const, rave_const_1]
    }
    player2 = {
        "name": "rave2",
        "run string": "python3 ../agents/mcts_agent.py r={rave_const} e={explore_const}".format(rave_const=rave_const_2, explore_const=explore_const),
        "hyperparameters": [explore_const, rave_const_2]
    }

    g = Game(
        board_size=11,
        player1=player1, player2=player2,
        verbose=True,
        log=True,
        print_protocol=True,
        kill_bots=True,
        silent_bots=False
    )
    g.run()

if __name__ == "__main__":
    main()