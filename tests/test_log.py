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

    def test_file_handler_when_enabled(self, tmp_path, monkeypatch):
        log_file = tmp_path / "predictor.log"
        monkeypatch.setenv("FOOTBALL_PREDICTOR__logging__file", str(log_file))
        logger = get_logger("test.file_handler")
        next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
        logger.info("file log message")
        assert log_file.exists()
        assert "file log message" in log_file.read_text()

    def test_default_format_fallback(self, monkeypatch):
        monkeypatch.delenv("FOOTBALL_PREDICTOR__logging__format", raising=False)
        logger = get_logger("test.default_fmt")
        assert any(
            h.formatter and "%(asctime)s" in h.formatter._fmt for h in logger.handlers
        )

    def test_same_name_returns_cached(self):
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2

    def test_can_log_without_error(self):
        logger = get_logger("test.can_log")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
