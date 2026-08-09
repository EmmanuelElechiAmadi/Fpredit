"""
Unit tests for the logging module.
"""

import logging

from src.log import get_logger


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test.logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.logger"

    def test_logger_propagate_disabled(self):
        logger = get_logger("test.no_propagate")
        assert logger.propagate is False

    def test_logger_has_console_handler(self):
        logger = get_logger("test.with_handler")
        handlers = logger.handlers
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)

    def test_same_name_returns_cached(self):
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2

    def test_can_log_without_error(self):
        logger = get_logger("test.can_log")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
