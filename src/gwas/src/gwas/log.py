# -*- coding: utf-8 -*-
import logging
import warnings
from logging.handlers import QueueHandler
from multiprocessing import get_context
from multiprocessing.queues import Queue
from pathlib import Path
from threading import Thread
from typing import TextIO

logger = logging.getLogger("gwas")
multiprocessing_context = get_context("spawn")


class LoggingThread(Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.logging_queue: Queue[logging.LogRecord] = Queue(ctx=multiprocessing_context)

    def run(self) -> None:
        while True:
            record = self.logging_queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)


logging_thread: LoggingThread | None = None


def _showwarning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    logger = logging.getLogger("py.warnings")
    logger.warning(
        warnings.formatwarning(message, category, filename, lineno, line),
        stack_info=True,
    )


def setup_logging(
    level: str | int, path: Path | None = None, stream: bool = True
) -> None:
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)8s] %(funcName)s: "
        "%(message)s (%(filename)s:%(lineno)s)"
    )

    handlers: list[logging.Handler] = []
    if stream is True:
        handlers.append(logging.StreamHandler())
    if path is not None:
        handlers.append(
            logging.FileHandler(path / "log.txt", "a", errors="backslashreplace")
        )
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    warnings.showwarning = _showwarning

    global logging_thread
    logging_thread = LoggingThread()
    logging_thread.start()

    logger.debug(f"Configured logging with handlers {handlers}")


def worker_configurer(
    logging_queue: Queue[logging.LogRecord], log_level: int | str
) -> None:
    queue_handler = QueueHandler(logging_queue)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(log_level)

    logging.captureWarnings(True)
