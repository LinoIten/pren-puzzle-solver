import sys
from pathlib import Path

# Parse and remove our flags before Kivy intercepts sys.argv
_regenerate = "--regenerate" in sys.argv
if _regenerate:
    sys.argv.remove("--regenerate")

_six_pieces = "--six-pieces" in sys.argv
if _six_pieces:
    sys.argv.remove("--six-pieces")

from src.core.pipeline import PuzzlePipeline
from src.core.config import Config
from src.utils.logger import setup_logger

# Projekt-Root zum Path hinzufügen
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():

    # Logger initialisieren
    logger = setup_logger("main")
    logger.info("=" * 60)
    logger.info("PREN Puzzle Solver gestartet")
    logger.info("=" * 60)

    try:

        config = Config()
        if _regenerate:
            config.vision.regenerate_mock = True
        if _six_pieces:
            config.vision.num_cuts = 3
        pipeline = PuzzlePipeline(config, show_ui=True)  # Enable UI
        result = pipeline.run()
        
        if result.success:
            logger.info("✓ Puzzle erfolgreich gelöst!")
            logger.info(f"Zeit: {result.duration:.2f}s")
        else:
            logger.error("✗ Puzzle konnte nicht gelöst werden")
            
    except KeyboardInterrupt:
        logger.info("\nProgramm durch Benutzer abgebrochen")
    except Exception as e:
        logger.exception(f"Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
