from MiniMax.CheckersGame.GameSettings import *

# count difference between the number of pieces, king+10
def basic_ev_func(board, is_black_turn):
    h = 0
    # funkcja liczy i zwraca ocene aktualnego stanu planszy
    # NEW: Basic evaluation function implemented by Mateusz Kolacz:
    if board.white_won:
        h -= WON_PRIZE
    if board.black_won:
        h += WON_PRIZE

    for row in board.board:
        for field in row:
            if field.is_black():
                if field.is_king():
                    h += KING_VALUE
                else:
                    h += PAWN_VALUE
            elif field.is_white():
                if field.is_king():
                    h -= KING_VALUE
                else:
                    h -= PAWN_VALUE

    # self.board[row][col].is_blue() - sprawdza czy to niebieski kolor figury
    # self.board[row][col].is_white()- sprawdza czy to biały kolor figury
    # self.board[row][col].is_king()- sprawdza czy to damka
    # self.board[row][col].row - wiersz na którym stoi figura
    # self.board[row][col].col - kolumna na której stoi figura
    # współrzędne zaczynają (0,0) się od lewej od góry
    return h


# nagrody jak w wersji podstawowej + nagroda za stopień zwartości grupy
def group_prize_ev_func(board, is_black_turn):
    h = 0
    # NEW: Basic + Group prize evaluation function implemented by Mateusz Kolacz:
    if board.white_won:
        h -= WON_PRIZE
    if board.black_won:
        h += WON_PRIZE
    """ example board state:
    board = [
        [-, b, -, W, -, b, -, W],
        [b, -, -, -, -, -, -, -],
        [-, b, -, -, -, -, -, b],
        [-, -, -, -, -, -, -, -],
        [-, -, -, b, -, w, -, -],
        [w, -, w, -, -, -, -, -],
        [-, w, -, -, -, -, -, w],
        [w, -, B, -, -, -, B, -]
    ]

    In my interpretation and implementation group density is calculated as follows:
        1. +/- 1 prize for each pawn/king neighbor in the same color 
            or if a pawn/king is next to a board's edge,
        2. the neighborhood is understood in such way that pawn/king 
            has in diagonal direction an another pawn/king of the same color standing next to them.

    example vicinity:
        vicinity = [
            [b, -, b],
            [-, b, -],
            [b, -, b]
        ]

    For the middle piece with coords: (row, col), the neighbors list looks as follows:
        neighbors_list = [ 
            vicinity[row - 1][col - 1], 
            vicinity[row - 1][col + 1], 
            vicinity[row + 1][col - 1],
            vicinity[row + 1][col + 1]
        ]
    """
    FIRST_ROW, FIRST_COL = 0, 0
    LAST_ROW, LAST_COL = 7, 7
    board_field = board.board

    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if board_field[row][col].is_black():
                if board_field[row][col].is_king():
                    h += KING_VALUE
                else:
                    h += PAWN_VALUE
                # Checking if a piece is next to board's edge and calculating bonus:
                if row == FIRST_ROW or row == LAST_ROW or col == FIRST_COL or col == LAST_COL:
                    h += 1

                # Calculating piece neighbors:
                # MAX one neighbor cases:
                # Top left corner case
                if row == FIRST_ROW and col == FIRST_COL:
                    if board_field[row + 1][col + 1].is_black():
                        h += 1
                        continue
                # Top right corner case
                if row == FIRST_ROW and col == LAST_COL:
                    if board_field[row + 1][col - 1].is_black():
                        h += 1
                        continue
                # Bottom left corner case
                if row == LAST_ROW and col == FIRST_COL:
                    if board_field[row - 1][col + 1].is_black():
                        h += 1
                        continue
                # Bottom right corner case
                if row == LAST_ROW and col == LAST_COL:
                    if board_field[row - 1][col - 1].is_black():
                        h += 1
                        continue

                # MAX two neighbors cases:
                # Top edge:
                if row == FIRST_ROW and (not col == FIRST_COL and not col == LAST_COL):
                    if board_field[row + 1][col - 1].is_black():
                        h += 1
                    if board_field[row + 1][col + 1].is_black():
                        h += 1
                    continue
                # Left edge:
                if col == FIRST_COL and (not row == FIRST_ROW and not row == LAST_ROW):
                    if board_field[row - 1][col + 1].is_black():
                        h += 1
                    if board_field[row + 1][col + 1].is_black():
                        h += 1
                    continue
                # Right edge:
                if col == LAST_COL and (not row == FIRST_ROW and not row == LAST_ROW):
                    if board_field[row + 1][col - 1].is_black():
                        h += 1
                    if board_field[row - 1][col - 1].is_black():
                        h += 1
                    continue
                # Bottom edge:
                if row == LAST_ROW and (not col == FIRST_COL and not col == LAST_COL):
                    if board_field[row - 1][col - 1].is_black():
                        h += 1
                    if board_field[row - 1][col + 1].is_black():
                        h += 1
                    continue

                # MAX four neighbors cases:
                if not col == FIRST_COL and not col == LAST_COL:
                    if board_field[row - 1][col - 1].is_black():
                        h += 1
                    if board_field[row - 1][col + 1].is_black():
                        h += 1
                    if board_field[row + 1][col - 1].is_black():
                        h += 1
                    if board_field[row + 1][col + 1].is_black():
                        h += 1
                    continue

            elif board_field[row][col].is_white():
                if board_field[row][col].is_king():
                    h -= KING_VALUE
                else:
                    h -= PAWN_VALUE
                # checking if a piece is next to board's edge and calculating bonus:
                if row == FIRST_ROW or row == LAST_ROW or col == FIRST_COL or col == LAST_COL:
                    h -= 1

                # Calculating piece neighbors:
                # MAX one neighbor cases:
                # Top left corner case
                if row == FIRST_ROW and col == FIRST_COL:
                    if board_field[row + 1][col + 1].is_white():
                        h -= 1
                        continue
                # Top right corner case
                if row == FIRST_ROW and col == LAST_COL:
                    if board_field[row + 1][col - 1].is_white():
                        h -= 1
                        continue
                # Bottom left corner case
                if row == LAST_ROW and col == FIRST_COL:
                    if board_field[row - 1][col + 1].is_white():
                        h -= 1
                        continue
                # Bottom right corner case
                if row == LAST_ROW and col == LAST_COL:
                    if board_field[row - 1][col - 1].is_white():
                        h -= 1
                        continue

                # MAX two neighbors cases:
                # Top edge:
                if row == FIRST_ROW and (not col == FIRST_COL and not col == LAST_COL):
                    if board_field[row - 1][col - 1].is_white():
                        h -= 1
                    if board_field[row - 1][col + 1].is_white():
                        h -= 1
                    continue
                # Left edge:
                if col == FIRST_COL and (not row == FIRST_ROW and not row == LAST_ROW):
                    if board_field[row - 1][col + 1].is_white():
                        h -= 1
                    if board_field[row + 1][col + 1].is_white():
                        h -= 1
                    continue
                # Right edge:
                if col == LAST_COL and (not row == FIRST_ROW and not row == LAST_ROW):
                    if board_field[row + 1][col - 1].is_white():
                        h -= 1
                    if board_field[row - 1][col - 1].is_white():
                        h -= 1
                    continue
                # Bottom edge:
                if row == LAST_ROW and (not col == FIRST_COL and not col == LAST_COL):
                    if board_field[row - 1][col - 1].is_white():
                        h -= 1
                    if board_field[row - 1][col + 1].is_white():
                        h -= 1
                    continue

                # MAX four neighbors cases:
                if not col == FIRST_COL and not col == LAST_COL:
                    if board_field[row - 1][col - 1].is_white():
                        h -= 1
                    if board_field[row - 1][col + 1].is_white():
                        h -= 1
                    if board_field[row + 1][col - 1].is_white():
                        h -= 1
                    if board_field[row + 1][col + 1].is_white():
                        h -= 1
                    continue

    return h


# za każdy pion na własnej połowie planszy otrzymuje się 5 nagrody, na połowie przeciwnika 7, a za każdą damkę 10.
def push_to_opp_half_ev_func(board, is_black_turn):
    h = 0
    # NEW: Push to opponent half evaluation function implemented by Mateusz Kolacz:
    if board.white_won:
        h -= WON_PRIZE
    if board.black_won:
        h += WON_PRIZE

    for row in board.board:
        for field in row:
            if field.is_black():
                if field.is_king():
                    h += KING_VALUE
                elif field.row < BOARD_HEIGHT / 2:
                    h += 5
                else:
                    h += 7
            elif field.is_white():
                if field.is_king():
                    h -= KING_VALUE
                elif field.row > BOARD_HEIGHT / 2:
                    h -= 5
                else:
                    h -= 7
    return h


# za każdy nasz pion otrzymuje się nagrodę w wysokości: (5 + numer wiersza, na którym stoi pion) (im jest bliżej wroga tym lepiej), a za każdą damkę dodatkowe: 10.
def push_forward_ev_func(board, is_black_turn):
    h = 0
    # NEW: Push forward evaluation function implemented by Mateusz Kolacz:
    if board.white_won:
        h -= WON_PRIZE
    if board.black_won:
        h += WON_PRIZE

    for row in board.board:
        for field in row:
            if field.is_black():
                if field.is_king():
                    h += KING_VALUE
                else:
                    h += 5 + field.row  # ToDO check if correct
            elif field.is_white():
                if field.is_king():
                    h -= KING_VALUE
                else:
                    h -= 5 + (BOARD_WIDTH - (field.row - 1))  # ToDO check if correct
    return h