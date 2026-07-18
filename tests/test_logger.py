from unittest.mock import patch

import pytest

import config
from logger import logger


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
