def extract_last_move_from_board(board) -> tuple:
    '''
    Decides if it is advantageous to swap
    based on the previous move 

        Parameters:
                board (str): String representation of the board 
                            received after each message

        Returns:
                (tuple): Last move, which was the first one,
                        before swap was used
    '''
    lines = board.split(',')
    for current_line in range(0, len(lines)):
        if 'R' in lines['current_line']:
            return (current_line, lines[current_line].indexOf('R'))
    return (-1, -1)
