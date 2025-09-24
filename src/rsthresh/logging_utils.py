import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import logging

def log_separator(logger: logging.Logger,
                  title: str | None = None,
                  char: str = "~",
                  width: int = 80,
                  lines: int = 1,
                  level: int = logging.INFO) -> None:
    line = char * width
    for _ in range(lines):
        logger.log(level, line)
    if title:
        logger.log(level, f"{char} {title} {char}")
    for _ in range(lines):
        logger.log(level, line)

def setup_logger(log_dir: str | Path,
                 name: str = "rsthresh",
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG,
                 filename: str = "logs_report.log") -> logging.Logger:
    """Configure a package-level logger with console + rotating file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))

    # Clear existing handlers if re-called
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)

    fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.debug(f"Logger initialized. Console={logging.getLevelName(console_level)}, "
                 f"File={logging.getLevelName(file_level)}, Path={log_path}")
    return logger
