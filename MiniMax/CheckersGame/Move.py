class Move:
    def __init__(self, piece, dest_row, dest_col, captures=None):
        self.piece = piece
        self.dest_row = dest_row
        self.dest_col = dest_col
        self.captures = captures

    def __eq__(self, other):
        if other is None:
            return False
        return self.piece == other.piece and self.dest_row == other.dest_row and self.dest_col == other.dest_col and self.captures == other.captures

    def __str__(self):
        return "Move from r, c:" + str(self.piece.row) + ", " + str(self.piece.col) + ", to:" + str(
            self.dest_row) + ", " + str(self.dest_col) + ", " + str(id(self.piece))