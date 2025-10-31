from src.utils.pose import Pose

class PuzzlePiece:
    def __init__(self, pid: str, pick: Pose):
        self.id = pid
        self.pick_pose = pick
        self.place_pose: Pose | None = None
        self.confidence = 0.0

    def __repr__(self) -> str:
        return (f'Piece(id={self.id}, pick={self.pick_pose}, '
                f'place={self.place_pose}, conf={self.confidence:.2f})')
