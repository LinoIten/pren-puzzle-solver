import numpy as np
from src.solver.validation.scorer import PlacementScorer
from src.ui.simulator.guess_renderer import GuessRenderer
from .guess_generator import GuessGenerator

class BruteForceSolver:
    def __init__(self):
        self.generator = GuessGenerator(rotation_step=90)
        self.renderer = GuessRenderer()
        self.scorer = PlacementScorer()
    
    def solve(self, num_pieces: int, positions: list, piece_shapes: dict, target: np.ndarray):
        """
        Try all guesses and return best one.
        """
        guesses = self.generator.generate_guesses(num_pieces, positions) # type: ignore
        
        print(f"Testing {len(guesses)} guesses...")
        
        best_score = -float('inf')
        best_guess = None
        best_rendered = None
        
        for i, guess in enumerate(guesses):
            rendered = self.renderer.render(guess, piece_shapes)
            score = self.scorer.score(rendered, target)
            
            if score > best_score:
                best_score = score
                best_guess = guess
                best_rendered = rendered
            
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(guesses)}, best: {best_score:.2f}")
        
        return best_guess, best_rendered, best_score