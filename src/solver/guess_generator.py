from itertools import permutations, product
from typing import List, Tuple

class GuessGenerator:
    def __init__(self, rotation_step: int = 90):
        self.rotation_step = rotation_step
    
    def generate_guesses(self, num_pieces: int, positions: List[Tuple[float, float]]) -> List[List[dict]]:
        """
        Generate all placement combinations.
        Returns: List of guesses, each guess is a list of {piece_id, x, y, theta}
        """
        rotations = list(range(0, 360, self.rotation_step))  # [0, 90, 180, 270]
        piece_ids = list(range(num_pieces))
        
        all_guesses = []
        
        for piece_order in permutations(piece_ids):
            for rotation_combo in product(rotations, repeat=num_pieces):
                guess = []
                for piece_id, pos, theta in zip(piece_order, positions[:num_pieces], rotation_combo):
                    guess.append({
                        'piece_id': piece_id,
                        'x': pos[0],
                        'y': pos[1],
                        'theta': theta
                    })
                all_guesses.append(guess)
        
        return all_guesses