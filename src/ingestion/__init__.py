"""Data Ingestion Package."""
from .download_dataset import download_uci_dataset
from .ingest_raw import ingest_raw_data

__all__ = ["download_uci_dataset", "ingest_raw_data"]
