from unittest.mock import patch

import pytest

from tddrobo import config
from tddrobo.logger import logger


@pytest.fixture(autouse=True)
def reset_progress():
    logger.update_progress(0, 0)


def test_logger_info(capsys):
    logger.info("Hello world")
    captured = capsys.readouterr()
    assert captured.out.endswith("[TDD Robo] Hello world\n")


def test_logger_warning(capsys):
    logger.warning("Be careful")
    captured = capsys.readouterr()
    assert captured.out.endswith("[TDD Robo] ⚠️ Warning: Be careful\n")


def test_logger_error(capsys):
    logger.error("Failed")
    captured = capsys.readouterr()
    assert captured.out.endswith("[TDD Robo] 🚨 Error: Failed\n")


def test_logger_debug_verbose(capsys):
    with patch.object(config, "VERBOSE", True):
        logger.debug("Debug msg")
        captured = capsys.readouterr()
        assert captured.out.endswith("[TDD Robo] Debug msg\n")


def test_logger_debug_non_verbose(capsys):
    with patch.object(config, "VERBOSE", False):
        logger.debug("Debug msg hidden")
        captured = capsys.readouterr()
        assert captured.out == ""


def test_logger_raw(capsys):
    logger.info("Raw output line", raw=True)
    captured = capsys.readouterr()
    assert captured.out == "Raw output line\n"


def test_logger_already_prefixed(capsys):
    logger.info("[TDD Robo] Pre-prefixed message")
    captured = capsys.readouterr()
    assert captured.out.endswith("[TDD Robo] Pre-prefixed message\n")


def test_logger_progress_suffix(capsys):
    logger.update_progress(1, 10)
    logger.info("Step 1")
    captured = capsys.readouterr()
    assert captured.out.endswith("[TDD Robo] (1/10) Step 1\n")

    # Reset progress to verify normal output
    logger.update_progress(0, 0)
    logger.info("Step reset")
    captured = capsys.readouterr()
    assert captured.out.endswith("[TDD Robo] Step reset\n")


def test_logger_add_file_handler(tmp_path):
    import logging
    import os

    log_file = tmp_path / "test.log"
    log_file_str = str(log_file)

    # Count initial file handlers
    initial_handlers = [h for h in logger._logger.handlers if isinstance(h, logging.FileHandler)]

    # Add file handler
    logger.add_file_handler(log_file_str)

    # Verify handler is added
    handlers_after = [h for h in logger._logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(handlers_after) == len(initial_handlers) + 1

    # Log some messages
    logger.info("Hello file log")
    logger.warning("Warning file log")

    # Flush handlers to ensure output is written
    for h in logger._logger.handlers:
        h.flush()

    # Read log file and check contents
    assert os.path.exists(log_file_str)
    with open(log_file_str, "r", encoding="utf-8") as f:
        content = f.read()

    assert "[TDD Robo] Hello file log" in content
    assert "[TDD Robo] ⚠️ Warning: Warning file log" in content

    # Add the same file handler again to verify duplicate prevention
    logger.add_file_handler(log_file_str)
    handlers_after_dup = [h for h in logger._logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(handlers_after_dup) == len(handlers_after)

    # Cleanup
    for h in list(logger._logger.handlers):
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file_str):
            h.close()
            logger._logger.removeHandler(h)


def test_logger_file_handler_append(tmp_path):
    import logging
    import os

    log_file = tmp_path / "append_test.log"
    log_file_str = str(log_file)

    # Write some initial data to the file
    with open(log_file_str, "w", encoding="utf-8") as f:
        f.write("Initial log line\n")

    # Add file handler
    logger.add_file_handler(log_file_str)

    # Log a new message
    logger.info("Appended log line")

    # Flush handlers
    for h in logger._logger.handlers:
        h.flush()

    # Read back and verify that it appended
    with open(log_file_str, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Initial log line\n" in content
    assert "[TDD Robo] Appended log line" in content

    # Cleanup
    for h in list(logger._logger.handlers):
        if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file_str):
            h.close()
            logger._logger.removeHandler(h)
