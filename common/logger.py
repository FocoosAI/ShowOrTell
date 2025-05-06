r"""Logging during training/testing"""

import datetime
import logging
import os

import torch


class DistributedLogger:
    def __init__(self, log_dir="logs", filename="log", rank=0):
        self.filename = filename
        self.rank = rank

        logtime = datetime.datetime.now().__format__("%m%d_%H%M%S")
        self.log_dir = log_dir + "_" + logtime
        os.makedirs(self.log_dir, exist_ok=True)

        # Use a unique logger name based on rank to avoid shared loggers between processes
        self.logger = logging.getLogger(f"process_{rank}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Prevent propagation to the root logger to avoid duplicate logging
        self.logger.propagate = False

        self._setup_handlers()

    def _setup_handlers(self):
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%d/%m %H:%M:%S",
        )

        # Create a separate log file for each rank
        log_file = os.path.join(self.log_dir, f"{self.filename}_rank{self.rank}.txt")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Only add stdout handler for the main process (rank 0)
        if self.rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    @staticmethod
    def is_main_process():
        return (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )

    def log_args(self, args):
        self.info(":======================= Arguments =========================")
        for arg_key in args.__dict__:
            self.info("| %20s: %-24s" % (arg_key, str(args.__dict__[arg_key])))
        self.info(":===========================================================")
