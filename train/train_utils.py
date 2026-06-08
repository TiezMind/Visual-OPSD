# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


class RobustFileHandler(logging.FileHandler):
    """FileHandler that auto-recovers when the log file or directory disappears.

    On NFS/NAS mounts or in long-running distributed jobs the log path can
    vanish transiently.  Instead of crashing the training loop, this handler
    recreates the directory & file and retries the write once.
    """

    def emit(self, record):
        try:
            super().emit(record)
        except FileNotFoundError:
            self._reopen()
            super().emit(record)

    def _reopen(self):
        log_dir = os.path.dirname(self.baseFilename)
        os.makedirs(log_dir, exist_ok=True)
        if self.stream and not self.stream.closed:
            self.stream.close()
        self.stream = self._open()


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                RobustFileHandler(f"{logging_dir}/{filename}.txt"),
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir
