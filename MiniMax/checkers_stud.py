#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.
Mam świadomość wielu jego braków ale nie mam czasu na jego poprawianie.

Zasady gry: https://en.wikipedia.org/wiki/English_draughts (w skrócie: wszyscy ruszają się po 1 polu. Pionki tylko w kierunku wroga, damki w dowolnym)
  z następującymi modyfikacjami: a) bicie nie jest wymagane,  b) dozwolone jest tylko pojedyncze bicie (bez serii).

Należy napisać funkcje "minimax_a_b_recurr", "minimax_a_b" (która woła funkcję rekurencyjną) i funkcje "*ev_func", która oceniają stan gry

Chętni mogą ulepszać mój kod (trzeba oznaczyć komentarzem co zostało zmienione), mogą również dodać obsługę bicia wielokrotnego i wymagania bicia. Mogą również wdrożyć reguły: https://en.wikipedia.org/wiki/Russian_draughts
"""

import numpy as np
import pygame # NOTE: Copied to Game.py
from copy import deepcopy # NOTE: Copied to Board.py

# NEW Game components imports:
from MiniMax.CheckersGame.Board import Board
from MiniMax.CheckersGame.Game import Game

#from CheckersGame.GameSettings import * # NEW importing constants/settings from separate file
from MiniMax.CheckersGame.GameSettings import *

# importing evaluation functions
from MiniMax.EvaluationFunctions.EvaluationFunctions import basic_ev_func, push_forward_ev_func, \
    push_to_opp_half_ev_func, group_prize_ev_func

""" NEW Exported below configuration to GameSettings.py

# FPS = 20
#
# MINIMAX_DEPTH = 5
#
# WIN_WIDTH = 800
# WIN_HEIGHT = 800
#
# WON_PRIZE = 10000
#
# MOVES_HIST_LEN=6
#
# BOARD_WIDTH = BOARD_HEIGHT = 8
#
# FIELD_SIZE = WIN_WIDTH/BOARD_WIDTH
# PIECE_SIZE = FIELD_SIZE/2 - 8
# MARK_THICK = 2
# POS_MOVE_MARK_SIZE = PIECE_SIZE/2
#
#
# BLACK_PIECES_COL = (0,0,0)
# WHITE_PIECES_COL = (255,255,255)
# POSS_MOVE_MARK_COL = (255,0,0)
# DARK_BOARD_COL = (196, 164, 132)
# BRIGHT_BOARD_COL = (250,250,250)
# KING_MARK_COL = (255, 215, 0)
"""


### NEW Evaluation functions have been exported to separate python file ###


#f. called from main    
def minimax_a_b(board, depth, plays_as_black, ev_func):
    possible_moves = board.get_possible_moves(plays_as_black)
    if len(possible_moves)==0:
        board.white_won = plays_as_black
        board.is_running=False
        return None
        
    a = -np.inf
    b = np.inf
    moves_marks = []
    for possible_move in possible_moves:
        # NEW: minimax_a_b function implemented by Mateusz Kolacz:
        board_copy = deepcopy(board)
        board_copy.make_move(possible_move) # ToDo Not sure if correct... Check it out once again
        moves_marks.append(minimax_a_b_recurr(
            board_copy,
            depth-1,
            not plays_as_black,
            a,
            b,
            ev_func
        ))

    if plays_as_black:
        best_index = moves_marks.index(max(moves_marks))
    else:
        best_index = moves_marks.index(min(moves_marks))

    return possible_moves[best_index]


#recursive function, called from minimax_a_b
def minimax_a_b_recurr(board, depth, move_max, a, b, ev_func):
    # NEW: Recursive function implemented by Mateusz Kolacz:
    if depth == 0 or (board.black_won or board.white_won):
        return ev_func(board, move_max)
    U = successors(board, move_max)

    if move_max:
        for u in U:
            a = max(a, minimax_a_b_recurr(u, depth - 1, not move_max, a, b, ev_func))
            if a >= b:
                return b
        return a
    else:
        for u in U:
            b = min(b, minimax_a_b_recurr(u, depth - 1, not move_max, a, b, ev_func))
            if a >= b:
                return a
        return b

# New: function calculating successor boards of the actual position
def successors(board, move_max):
    successors_boards = []
    for move in board.get_possible_moves(move_max):
        board_copy = deepcopy(board)
        board_copy.make_move(move)
        successors_boards.append(board_copy)
    return successors_boards

# NEW Exported to separate file...
"""
class Move:
    def __init__(self, piece, dest_row, dest_col, captures=None):
        self.piece=piece
        self.dest_row=dest_row
        self.dest_col=dest_col
        self.captures=captures
        
    def __eq__(self, other):
        if other is None:
            return False
        return self.piece==other.piece and self.dest_row==other.dest_row and self.dest_col==other.dest_col and self.captures==other.captures
    
    def __str__(self):
        return "Move from r, c:"+str(self.piece.row)+", "+str(self.piece.col)+", to:"+str(self.dest_row)+", "+ str(self.dest_col)+", "+ str(id(self.piece))
"""

# NEW Exported to separate file...
"""
class Field:
    def is_empty(self):
        return True
    
    def is_white(self):
        return False

    def is_black(self):
        return False
    
    def __str__(self):
        return "."   



# NEW Exported to separate file...
"""

# NEW Exported to separate file...
"""
class Pawn(Field):    
    def __init__(self, is_white, row, col):
        self.__is_white=is_white
        self.row = row
        self.col = col
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result  

    def __str__(self):
        if self.is_white():
            return "w"
        return "b"
            
    def is_king(self):
        return False
    
    def is_empty(self):
        return False
    
    def is_white(self):
        return self.__is_white

    def is_black(self):
        return not self.__is_white
"""

# NEW Exported to separate file...
"""
class King(Pawn):    
    def __init__(self, pawn):
        super().__init__(pawn.is_white(), pawn.row, pawn.col)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result  
    
    def is_king(self):
        return True

    def __str__(self):
        if self.is_white():
            return "W"
        return "B"
"""

# NEW Exported to separate file...
"""
class Board:
    def __init__(self): #row, col
        self.board = []
        self.white_turn = True
        self.white_fig_left = 12
        self.black_fig_left = 12
        self.black_won = False
        self.white_won = False
        self.capture_exists = False
        self.last_white_mov_indx=0
        self.white_moves_hist=[None]*MOVES_HIST_LEN
        self.black_moves_hist=[None]*MOVES_HIST_LEN
        self.last_black_mov_indx=0
        self.black_repeats=False
        self.white_repeats=False

        self.__set_pieces()
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        result.board= deepcopy(self.board )
        return result  

    def __str__(self):
        to_ret=""
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                to_ret+=str(self.board[row][col])
            to_ret+="\n"
        return to_ret

    #useful only for debugging (set board according to given list of strings)
    def set(self, b): 
        self.white_fig_left = 0
        self.black_fig_left = 0
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                fig=Field()
                if b[row][col]=="b" or b[row][col]=="w":
                    fig = Pawn(b[row][col]=="w", row, col)
                    
                if b[row][col]=="B" or b[row][col]=="W":
                    fig = King(Pawn(b[row][col]=="W", row, col))
                
                self.board[row][col]=fig
                if self.board[row][col].is_black():
                    self.black_fig_left+=1
                if self.board[row][col].is_white():
                    self.white_fig_left+=1
                
    #initializes board    
    def __set_pieces(self):
        for row in range(BOARD_HEIGHT):
            self.board.append([])
            for col in range(BOARD_WIDTH):
                self.board[row].append( Field() )

        for row in range(BOARD_HEIGHT//2-1):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(False, row, col)

        for row in range(BOARD_HEIGHT//2+1, BOARD_HEIGHT):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(True, row, col)

    #get possible moves for piece
    def get_piece_moves(self, piece):
        pos_moves=[]
        row = piece.row
        col = piece.col
        if piece.is_black():
            enemy_is_white = True
        else:
            enemy_is_white = False

        if piece.is_white() or (piece.is_black() and piece.is_king()):            
            dir_y = -1
            if row > 0:
                new_row=row+dir_y
                if col > 0:
                    new_col=col-1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))  
                    #captures
                    elif self.board[new_row][new_col].is_white()==enemy_is_white and new_row+dir_y>=0 and new_col-1>=0 and self.board[new_row+dir_y][new_col-1].is_empty():
                        pos_moves.append(Move(piece, new_row+dir_y, new_col-1, self.board[new_row][new_col]))  
                        self.capture_exists = True
                        
                if col < BOARD_WIDTH-1:
                    new_col=col+1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece,new_row, new_col))    
                    #captures
                    elif self.board[new_row][new_col].is_white()==enemy_is_white and new_row+dir_y>=0 and new_col+1<BOARD_WIDTH and self.board[new_row+dir_y][new_col+1].is_empty():
                        pos_moves.append(Move(piece,new_row+dir_y, new_col+1, self.board[new_row][new_col]))  
                        self.capture_exists = True

        if piece.is_black() or (piece.is_white() and self.board[row][col].is_king()):
            dir_y = 1
            if row<BOARD_WIDTH-1:
                new_row=row+dir_y
                if col > 0:
                    new_col=col-1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece,new_row, new_col))    
                    elif self.board[new_row][new_col].is_white()==enemy_is_white and new_row+dir_y<BOARD_WIDTH and new_col-1>=0 and self.board[new_row+dir_y][new_col-1].is_empty():
                        pos_moves.append(Move(piece,new_row+dir_y, new_col-1, self.board[new_row][new_col]))  
                        self.capture_exists = True
                        
                if col < BOARD_WIDTH-1:
                    new_col=col+1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece,new_row, new_col))    
                    #captures
                    elif self.board[new_row][new_col].is_white()==enemy_is_white and new_row+dir_y<BOARD_WIDTH and new_col+1<BOARD_WIDTH and self.board[new_row+dir_y][new_col+1].is_empty():
                        pos_moves.append(Move(piece,new_row+dir_y, new_col+1, self.board[new_row][new_col]))  
                        self.capture_exists = True
        return pos_moves
    
          
    #get possible moves for player
    def get_possible_moves(self, is_black_turn):
        pos_moves = []
        self.capture_exists = False
        for row in range(BOARD_WIDTH):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_black_turn and self.board[row][col].is_black()) or (not is_black_turn and self.board[row][col].is_white()):                        
                        pos_moves.extend( self.get_piece_moves(self.board[row][col]) )
        return pos_moves
        
                
    #detect draws                        
    def end(self):
        #stop if repeats
        if self.black_repeats and self.white_repeats:
            #who won
            ev=basic_ev_func(self, not self.white_turn)
            if ev>0:
                self.black_won=True
            elif ev<0:
                self.white_won=True
            else:
                self.black_won=True
                self.white_won=True
            return True
        return False
    
    #used for useless play detection (game is stopped when players repeats the same moves)
    def register_move(self, move):
        move_tuple = (move.piece.row, move.piece.col, move.dest_row, move.dest_col, id(move.piece))

        if self.white_turn:
            self.white_repeats=False
            if move_tuple in self.white_moves_hist:
                self.white_repeats = True
            self.white_moves_hist[self.last_white_mov_indx] = move_tuple
            self.last_white_mov_indx += 1
            if self.last_white_mov_indx >= MOVES_HIST_LEN:
                self.last_white_mov_indx=0
        else:
            self.black_repeats=False
            if move_tuple in self.black_moves_hist:
                self.black_repeats=True
            self.black_moves_hist[self.last_black_mov_indx] = move_tuple
            self.last_black_mov_indx += 1
            if self.last_black_mov_indx >= MOVES_HIST_LEN:
                self.last_black_mov_indx=0

    #execute move on board
    def make_move(self, move):  
        d_row = move.dest_row
        d_col = move.dest_col
        row_from = move.piece.row
        col_from = move.piece.col

        self.board[d_row][d_col]=self.board[row_from][col_from]
        self.board[d_row][d_col].row=d_row
        self.board[d_row][d_col].col=d_col
        self.board[row_from][col_from]=Field()     

        if move.captures:
            fig_to_del = move.captures
            self.board[fig_to_del.row][fig_to_del.col]=Field()
            if self.white_turn:
                self.black_fig_left -= 1
            else:
                self.white_fig_left -= 1
            
        if self.white_turn and d_row==0 and not self.board[d_row][d_col].is_king():#damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if not self.white_turn and d_row==BOARD_WIDTH-1 and not self.board[d_row][d_col].is_king():#damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])
            
        self.white_turn = not self.white_turn
"""

# NEW Exported to separate file...
"""
class Game:
    def __init__(self, window, board):
        self.window = window
        self.board = board
        self.something_is_marked = False
        self.marked_col = None
        self.marked_row = None
        self.pos_moves={}
        
        
    def __draw(self):
        self.window.fill(BRIGHT_BOARD_COL)
        #draw board
        for row in range(BOARD_HEIGHT):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                y = row*FIELD_SIZE
                x = col*FIELD_SIZE
                pygame.draw.rect(self.window, DARK_BOARD_COL, (x, y , FIELD_SIZE, FIELD_SIZE))

        #draw pieces
        for row in range(BOARD_HEIGHT):
            for col in range((row+1) % 2, BOARD_WIDTH, 2):
                cur_col = None
                if self.board.board[row][col].is_white():
                    cur_col = WHITE_PIECES_COL
                elif self.board.board[row][col].is_black():
                    cur_col = BLACK_PIECES_COL
                if cur_col is not None:                    
                    x = col*FIELD_SIZE
                    y = row*FIELD_SIZE
                    pygame.draw.circle(self.window, cur_col, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE)
                    if self.board.board[row][col].is_king(): 
                        pygame.draw.circle(self.window, KING_MARK_COL, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE/2)
    
        #if piece is marked by user, mark it and possible moves
        if self.something_is_marked:
            x = self.marked_col*FIELD_SIZE
            y = self.marked_row*FIELD_SIZE
            pygame.draw.circle(self.window, POSS_MOVE_MARK_COL, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), PIECE_SIZE+MARK_THICK, MARK_THICK)
            pos_moves = self.board.get_piece_moves(self.board.board[self.marked_row][self.marked_col])
            for pos_move in pos_moves:
                self.pos_moves[(pos_move.dest_row,pos_move.dest_col)] = pos_move
                x = pos_move.dest_col*FIELD_SIZE
                y = pos_move.dest_row*FIELD_SIZE
                pygame.draw.circle(self.window, POSS_MOVE_MARK_COL, (x+FIELD_SIZE/2, y+FIELD_SIZE/2), POS_MOVE_MARK_SIZE)           

    def update(self):
        self.__draw()
        pygame.display.update()
    
    def mouse_to_indexes(self, pos):
        return (int(pos[0]//FIELD_SIZE), int(pos[1]//FIELD_SIZE))

    def clicked_at(self, pos):
        (col, row) = self.mouse_to_indexes(pos)
        field = self.board.board[row][col]
        if self.something_is_marked:
            if (row, col) in self.pos_moves:
                self.board.make_move(self.pos_moves[(row, col)])
                self.something_is_marked = False
                self.pos_moves={}


        if field.is_white():
            if self.something_is_marked:
                self.something_is_marked = False
                self.pos_moves={}
            else:
                self.something_is_marked = True
                self.marked_col=col
                self.marked_row=row
"""

def main():
    board = Board()
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    is_running = True
    clock = pygame.time.Clock()
    game = Game(window, board)

    while is_running:
        clock.tick(FPS)
            
        if not game.board.white_turn:
            move = minimax_a_b( game.board, MINIMAX_DEPTH, True, basic_ev_func)
            #move = minimax_a_b( game.board, MINIMAX_DEPTH, True, push_forward_ev_func)
            #move = minimax_a_b( game.board, MINIMAX_DEPTH, True, push_to_opp_half_ev_func)
            #move = minimax_a_b( game.board, MINIMAX_DEPTH, True, group_prize_ev_func)
            
            
            
            if move is not None:
                game.board.make_move(move)
            else:
                is_running = False
        if game.board.end():
            is_running = False


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.clicked_at(pos)

        game.update()

    pygame.quit()


def ai_vs_ai():
    board = Board()
    is_running = True
 
    while is_running:
        if board.white_turn:
            move = minimax_a_b( board, 5, not board.white_turn, basic_ev_func)
        else:
            move = minimax_a_b( board, 5, not board.white_turn, basic_ev_func)
            # move = minimax_a_b( board, 5, not board.white_turn, push_forward_ev_func)
            # move = minimax_a_b( board, 5, not board.white_turn, push_to_opp_half_ev_func)
            # move = minimax_a_b( board, 5, not board.white_turn, group_prize_ev_func)
            
        if move is not None:
            board.register_move(move)
            board.make_move(move)
        else:
            if board.white_turn:
                board.black_won=True
            else:
                board.white_won=True
            is_running = False
        if board.end():
            is_running = False
    print("black_won:", board.black_won )
    print("white_won:", board.white_won )
    #if both won then it is a draw!
    

main()
#ai_vs_ai()
    

