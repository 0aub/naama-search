#!/usr/bin/env python3
"""
Enhanced logging system for Naama Search
Based on the original notebook's DatedLogger implementation
"""

import os
import time
import datetime
import logging
import pathlib
from logging.handlers import BaseRotatingHandler
from rich.logging import RichHandler
from typing import Union, Optional


# Add TRACE level like original notebook
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")

def _trace(self: logging.Logger, msg: str, *args, **kw):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kw)

logging.Logger.trace = _trace


class DatedLogger:
    """
    Enhanced logging system matching original notebook
    
    Features:
    - Daily rotating log files with YYYY-MM-DD suffix
    - Rich console output with colors
    - Configurable log levels for console and file
    - Proper formatting and timestamps
    - HuggingFace library integration
    
    Usage:
        from src.logger import DatedLogger
        log = DatedLogger("preset-testing").logger
        log.info("Test message")
    """

    class _FileHandler(BaseRotatingHandler):
        """Custom file handler that rotates daily with date suffix"""
        
        def __init__(self, base: pathlib.Path, utc: bool, level: int):
            self.base = pathlib.Path(base)
            self.utc = utc
            filename = f"{self.base}.{self._today()}"
            super().__init__(filename, mode="a", encoding="utf-8")
            self.setLevel(level)
            self.rollover_at = self._next_midnight()

        def _now(self):
            return datetime.datetime.utcnow() if self.utc else datetime.datetime.now()
            
        def _today(self):
            return self._now().strftime("%Y-%m-%d")
            
        def _next_midnight(self):
            nxt = (self._now() + datetime.timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            return time.mktime(nxt.timetuple())

        def shouldRollover(self, record): 
            return time.time() >= self.rollover_at
            
        def doRollover(self):
            if self.stream: 
                self.stream.close()
            self.baseFilename = f"{self.base}.{self._today()}"
            self.stream = self._open()
            self.rollover_at = self._next_midnight()

    def __init__(
        self, 
        name: str = "naama-search", 
        log_dir: Union[str, pathlib.Path] = "logs",
        console_level: int = logging.INFO, 
        file_level: int = logging.DEBUG,
        utc: bool = False, 
        hf_verbose: bool = False, 
        force_new: bool = False
    ) -> None:
        """
        Initialize the logging system
        
        Args:
            name: Logger name (will be used for log filename)
            log_dir: Directory to store log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            utc: Use UTC timestamps instead of local time
            hf_verbose: Enable verbose HuggingFace logging
            force_new: Force creation of new logger even if exists
        """
        
        log_dir = pathlib.Path(log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        if logger.handlers and not force_new:
            self.logger = logger
            return
            
        logger.handlers.clear()

        # File handler with daily rotation
        base_file = log_dir / f"{name}.log"
        fh = self._FileHandler(base_file, utc=utc, level=file_level)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s  %(levelname)-7s] %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

        # Rich console handler with colors
        ch = RichHandler(
            level=console_level, 
            markup=False, 
            show_level=True,
            show_time=True, 
            show_path=False, 
            rich_tracebacks=True
        )
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

        # Configure HuggingFace logging if requested
        if hf_verbose:
            for lib in ("transformers", "huggingface_hub"):
                sub = logging.getLogger(lib)
                sub.setLevel(logging.INFO)
                sub.handlers.clear()
                sub.addHandler(fh)
                sub.addHandler(ch)
                sub.propagate = False
            try:
                from transformers.utils import logging as txlog
                txlog.set_verbosity_info()
                from huggingface_hub.utils import logging as hublog
                hublog.set_verbosity_info()
            except ModuleNotFoundError:
                pass

        # Configure root logger to catch stray messages
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.handlers.clear()
        root.addHandler(fh)
        root.addHandler(ch)

        logger.info(f"📝 Logging to {fh.baseFilename}")
        self.logger = logger


# Convenience function for quick logger creation
def get_logger(
    name: str = "naama-search",
    log_dir: Union[str, pathlib.Path] = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    **kwargs
) -> logging.Logger:
    """
    Quick way to get a configured logger
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
        **kwargs: Additional arguments for DatedLogger
        
    Returns:
        Configured logger instance
    """
    return DatedLogger(
        name=name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level,
        **kwargs
    ).logger


# Global logger instance for convenience
default_logger = None

def init_default_logger(
    name: str = "naama-search",
    **kwargs
) -> logging.Logger:
    """
    Initialize and return a default logger for the application
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for DatedLogger
        
    Returns:
        Default logger instance
    """
    global default_logger
    default_logger = get_logger(name=name, **kwargs)
    return default_logger


def get_default_logger() -> Optional[logging.Logger]:
    """
    Get the default logger instance
    
    Returns:
        Default logger if initialized, None otherwise
    """
    return default_logger