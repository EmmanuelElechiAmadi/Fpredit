"""
Simple logging setup that integrates with config.yaml.

Usage:
    from src.log import get_logger
    log = get_logger(__name__)
    log.info("Model training started")
    log.warning("Elo not converging: %s", message)

The log level and format are controlled via config.yaml → logging section.
"""

import logging
import sys

from src.config import load_config

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger configured per config.yaml.

    Parameters
    ----------
    name : str
        Logger name, typically __name__.

    Returns
    -------
    logging.Logger
    """
    if name in _loggers:
        return _loggers[name]

    cfg = load_config()
    level = getattr(
        logging,
        (cfg.logging.level if hasattr(cfg, "logging") else "INFO").upper(),
        logging.INFO,
    )
    fmt = getattr(cfg.logging, "format", None) if hasattr(cfg, "logging") else None
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Always add a console handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Optional file handler
    if hasattr(cfg, "logging") and hasattr(cfg.logging, "file") and cfg.logging.file:
        from logging import FileHandler

        fh = FileHandler(cfg.logging.file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger
