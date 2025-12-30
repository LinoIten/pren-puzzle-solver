import sys
from pathlib import Path

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
