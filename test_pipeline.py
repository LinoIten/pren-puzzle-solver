
from src.core.config import Config
from src.core.pipeline import PuzzlePipeline

if __name__ == '__main__':
    print("=== Running Pipeline with UI ===")
    config = Config()
    pipeline = PuzzlePipeline(config, show_ui=True)  # Enable UI
    result = pipeline.run()
    
    print(f"\nResult: {result.message}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Success: {result.success}")

    