from MiniMax.CheckersGame.GameSettings import *
from MiniMax.CheckersGame.Field import Field
from MiniMax.CheckersGame.Pawn import Pawn
from MiniMax.CheckersGame.King import King
from MiniMax.CheckersGame.Move import Move
from MiniMax.EvaluationFunctions.EvaluationFunctions import basic_ev_func
from copy import deepcopy


class Board:
    def __init__(self):  # row, col
        self.board = []
        self.white_turn = True
        self.white_fig_left = 12
        self.black_fig_left = 12
        self.black_won = False
        self.white_won = False
        self.capture_exists = False
        self.last_white_mov_indx = 0
        self.white_moves_hist = [None] * MOVES_HIST_LEN
        self.black_moves_hist = [None] * MOVES_HIST_LEN
        self.last_black_mov_indx = 0
        self.black_repeats = False
        self.white_repeats = False

        self.__set_pieces()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        result.board = deepcopy(self.board)
        return result

    def __str__(self):
        to_ret = ""
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                to_ret += str(self.board[row][col])
            to_ret += "\n"
        return to_ret

    # useful only for debugging (set board according to given list of strings)
    def set(self, b):
        self.white_fig_left = 0
        self.black_fig_left = 0
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                fig = Field()
                if b[row][col] == "b" or b[row][col] == "w":
                    fig = Pawn(b[row][col] == "w", row, col)

                if b[row][col] == "B" or b[row][col] == "W":
                    fig = King(Pawn(b[row][col] == "W", row, col))

                self.board[row][col] = fig
                if self.board[row][col].is_black():
                    self.black_fig_left += 1
                if self.board[row][col].is_white():
                    self.white_fig_left += 1

    # initializes board
    def __set_pieces(self):
        for row in range(BOARD_HEIGHT):
            self.board.append([])
            for col in range(BOARD_WIDTH):
                self.board[row].append(Field())

        for row in range(BOARD_HEIGHT // 2 - 1):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(False, row, col)

        for row in range(BOARD_HEIGHT // 2 + 1, BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(True, row, col)

    # get possible moves for piece
    def get_piece_moves(self, piece):
        pos_moves = []
        row = piece.row
        col = piece.col
        if piece.is_black():
            enemy_is_white = True
        else:
            enemy_is_white = False

        if piece.is_white() or (piece.is_black() and piece.is_king()):
            dir_y = -1
            if row > 0:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # captures
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y >= 0 and new_col - 1 >= 0 and \
                            self.board[new_row + dir_y][new_col - 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col - 1, self.board[new_row][new_col]))
                        self.capture_exists = True

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # captures
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y >= 0 and new_col + 1 < BOARD_WIDTH and \
                            self.board[new_row + dir_y][new_col + 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col + 1, self.board[new_row][new_col]))
                        self.capture_exists = True

        if piece.is_black() or (piece.is_white() and self.board[row][col].is_king()):
            dir_y = 1
            if row < BOARD_WIDTH - 1:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y < BOARD_WIDTH and new_col - 1 >= 0 and \
                            self.board[new_row + dir_y][new_col - 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col - 1, self.board[new_row][new_col]))
                        self.capture_exists = True

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                        # captures
                    elif self.board[new_row][
                        new_col].is_white() == enemy_is_white and new_row + dir_y < BOARD_WIDTH and new_col + 1 < BOARD_WIDTH and \
                            self.board[new_row + dir_y][new_col + 1].is_empty():
                        pos_moves.append(Move(piece, new_row + dir_y, new_col + 1, self.board[new_row][new_col]))
                        self.capture_exists = True
        return pos_moves

    # get possible moves for player
    def get_possible_moves(self, is_black_turn):
        pos_moves = []
        self.capture_exists = False
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_black_turn and self.board[row][col].is_black()) or (
                            not is_black_turn and self.board[row][col].is_white()):
                        pos_moves.extend(self.get_piece_moves(self.board[row][col]))
        return pos_moves

    # detect draws
    def end(self):
        # stop if repeats
        if self.black_repeats and self.white_repeats:
            # who won
            ev = basic_ev_func(self, not self.white_turn)
            if ev > 0:
                self.black_won = True
            elif ev < 0:
                self.white_won = True
            else:
                self.black_won = True
                self.white_won = True
            return True
        return False

    # used for useless play detection (game is stopped when players repeats the same moves)
    def register_move(self, move):
        move_tuple = (move.piece.row, move.piece.col, move.dest_row, move.dest_col, id(move.piece))

        if self.white_turn:
            self.white_repeats = False
            if move_tuple in self.white_moves_hist:
                self.white_repeats = True
            self.white_moves_hist[self.last_white_mov_indx] = move_tuple
            self.last_white_mov_indx += 1
            if self.last_white_mov_indx >= MOVES_HIST_LEN:
                self.last_white_mov_indx = 0
        else:
            self.black_repeats = False
            if move_tuple in self.black_moves_hist:
                self.black_repeats = True
            self.black_moves_hist[self.last_black_mov_indx] = move_tuple
            self.last_black_mov_indx += 1
            if self.last_black_mov_indx >= MOVES_HIST_LEN:
                self.last_black_mov_indx = 0

    # execute move on board
    def make_move(self, move):
        d_row = move.dest_row
        d_col = move.dest_col
        row_from = move.piece.row
        col_from = move.piece.col

        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if move.captures:
            fig_to_del = move.captures
            self.board[fig_to_del.row][fig_to_del.col] = Field()
            if self.white_turn:
                self.black_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if self.white_turn and d_row == 0 and not self.board[d_row][d_col].is_king():  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row == BOARD_WIDTH - 1 and not self.board[d_row][d_col].is_king():  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn