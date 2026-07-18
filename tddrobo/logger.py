import builtins
import logging
import sys
import threading
import time

from tddrobo import config

_thread_local = threading.local()


class TDDRoboFormatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        # If record has a 'raw' attribute set to True, print verbatim
        if getattr(record, "raw", False):
            return msg

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        progress_suffix = ""
        current_req = getattr(_thread_local, "current_req_num", 0)
        total_req = getattr(_thread_local, "total_req_num", 0)
        if current_req > 0 and total_req > 0:
            progress_suffix = f" ({current_req}/{total_req})"

        # Default prefix
        prefix = f"[{timestamp}] [TDD Robo]{progress_suffix} "
        if record.levelno == logging.WARNING:
            prefix += "⚠️ Warning: "
        elif record.levelno == logging.ERROR:
            prefix += "🚨 Error: "

        if msg.startswith("[TDD Robo]"):
            # Strip the raw prefix to avoid duplication
            msg = msg[len("[TDD Robo]") :].strip()

        return f"{prefix}{msg}"


class StdoutProxy:
    def write(self, data):
        sys.stdout.write(data)

    def flush(self):
        sys.stdout.flush()


class TDDRoboLogger:
    def __init__(self):
        self._logger = logging.getLogger("tddrobo")
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            handler = logging.StreamHandler(StdoutProxy())
            handler.setFormatter(TDDRoboFormatter())
            self._logger.addHandler(handler)

    def add_file_handler(self, log_file_path: str):
        """Add a file handler to log outputs to a specific file."""
        import os

        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        abs_path = os.path.abspath(log_file_path)
        for h in list(self._logger.handlers):
            if isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == abs_path:
                return

        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(TDDRoboFormatter())
        self._logger.addHandler(file_handler)

    def update_progress(self, current: int, total: int):
        """Update the requirement progress context for the current thread."""
        _thread_local.current_req_num = current
        _thread_local.total_req_num = total

    def _format_args(self, *args) -> str:
        return " ".join(str(arg) for arg in args)

    def info(self, *args, raw=False):
        self._logger.info(self._format_args(*args), extra={"raw": raw})

    def warning(self, *args, raw=False):
        self._logger.warning(self._format_args(*args), extra={"raw": raw})

    def error(self, *args, raw=False):
        self._logger.error(self._format_args(*args), extra={"raw": raw})

    def debug(self, *args, raw=False):
        if getattr(config, "VERBOSE", False):
            self._logger.debug(self._format_args(*args), extra={"raw": raw})


# Singleton instance
logger = TDDRoboLogger()


def print(*args, **kwargs):
    # Bypass logging for dynamic spinners, stream output, or empty prints (newlines)
    if "end" in kwargs or "flush" in kwargs or not args or args == ("",):
        builtins.print(*args, **kwargs)
        return

    msg = " ".join(str(a) for a in args)
    if "⚠️" in msg or "Warning" in msg or "warning" in msg:
        logger.warning(msg)
    elif "🚨" in msg or "❌" in msg:
        logger.error(msg)
    else:
        if msg.startswith("\n") or "===" in msg or "```" in msg:
            logger.info(msg, raw=True)
        else:
            logger.info(msg)
