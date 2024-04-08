from MiniMax.CheckersGame.Pawn import Pawn


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