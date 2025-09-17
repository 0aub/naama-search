"""
Utility functions and classes: data loading, logging, configuration loading.
Combined utilities for better organization.
"""

import pathlib
import datetime
import time
import csv
import json
import yaml
import logging
from logging.handlers import BaseRotatingHandler
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from langchain.docstore.document import Document
from rich.logging import RichHandler

log = logging.getLogger(__name__)


def load_config(filename: str) -> dict:
    """Load configuration from config folder"""
    config_path = pathlib.Path(__file__).parent.parent / "config" / filename
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class DatedLogger:
    """
    Enhanced logger with dated file rotation and rich console output.
    
    Usage:
        log = DatedLogger("naama-search", hf_verbose=True).logger
        log.info("Hi!")
        log.trace("A very chatty line")
    """

    class _FileHandler(BaseRotatingHandler):
        """Custom file handler with daily rotation"""
        
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

    def __init__(self, name: str = "service-search", log_dir: str | pathlib.Path = "logs",
                 console_level: int = logging.INFO, file_level: int = logging.DEBUG,
                 utc: bool = False, hf_verbose: bool = False, force_new: bool = False) -> None:

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

        # Rich console handler
        ch = RichHandler(
            level=console_level, markup=False, show_level=True,
            show_time=True, show_path=False, rich_tracebacks=True
        )
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

        # HuggingFace logging
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

        # Root logger
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.handlers.clear()
        root.addHandler(fh)
        root.addHandler(ch)

        logger.info(f"📝 logging to {fh.baseFilename}")
        self.logger = logger


# Add TRACE level
TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")

def _trace(self: logging.Logger, msg: str, *args, **kw):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kw)

logging.Logger.trace = _trace


class ServiceDatasetLoader:
    """Loads datasets, remaps columns, builds LangChain Documents with text processing."""
    
    def __init__(self, file_path: str | pathlib.Path, rename_map: Dict[str, str],
                 combine_cols: Sequence[str], separator: str = " | ", label: str | None = None,
                 normalizer=None, morpher=None) -> None:
        
        self.path = pathlib.Path(file_path)
        self.label = label or self.path.stem
        self.rename_map = rename_map
        self.combine_cols = tuple(combine_cols)
        self.separator = separator
        self.normalizer = normalizer
        self.morpher = morpher

        log.info(f"📄 Loading [{self.label}] '{self.path.name}'…")
        self.df, self.documents = self._prepare()
        log.info(f"✅ {len(self.df):,} rows → ({', '.join(self.combine_cols)})")

    def _prepare(self) -> Tuple[pd.DataFrame, List[Document]]:
        """Prepare dataframe and create documents"""
        df = self._read_file(self.path)
        df = df.rename(columns=self.rename_map).rename(columns=str.lower)
        
        # Validate columns exist
        for c in self.combine_cols:
            if c not in df.columns:
                raise KeyError(f"Column '{c}' missing after rename")
            df[c] = df[c].astype(str).str.strip()
        
        # Clean and deduplicate
        subset_cols = list(dict.fromkeys(self.combine_cols))
        df = (df.drop_duplicates(subset=subset_cols)
               .dropna(subset=subset_cols)
               .reset_index(drop=True))
        
        # Combine text fields
        df["combined_text"] = df.apply(
            lambda r: self.separator.join(r[c] for c in self.combine_cols), axis=1
        )

        # Text normalization
        if self.normalizer:
            df["combined_text_norm"] = df["combined_text"].apply(
                lambda t: self.normalizer.normalize(t, self.label)
            )
        else:
            df["combined_text_norm"] = df["combined_text"]

        # Morphological reduction
        if self.morpher:
            df["combined_text_base"] = df["combined_text_norm"].apply(
                lambda t: self.morpher.reduce(t, self.label)
            )
        else:
            df["combined_text_base"] = df["combined_text_norm"]

        # Create documents
        docs = [
            Document(
                page_content=row["combined_text_base"],
                metadata={c: row[c] for c in self.combine_cols} | {
                    "combined_text_raw": row["combined_text"],
                    "combined_text_norm": row["combined_text_norm"],
                },
            )
            for _, row in df.iterrows()
        ]
        return df, docs

    @staticmethod
    def _read_file(path: pathlib.Path) -> pd.DataFrame:
        """Read file based on extension"""
        suffix = path.suffix.lower()
        
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, engine="openpyxl")
        
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return pd.json_normalize(obj) if isinstance(obj, list) else pd.DataFrame(obj)
        
        # CSV/TSV
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(2048)
            dialect = csv.Sniffer().sniff(sample)
            f.seek(0)
            return pd.read_csv(f, sep=dialect.delimiter, engine="python", on_bad_lines="skip")