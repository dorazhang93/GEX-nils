import os
from pathlib import Path

PROJECT_DIR = Path('/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/').absolute().resolve()
DATA_DIR = Path('/home/avesta/daqu/Projects/GEX/GEX_processed/modeling_data').absolute().resolve()
OUTPUT_DIR = PROJECT_DIR / 'output'


def get_path(relative_path: str) -> Path:
    return (
        Path(relative_path)
        if relative_path.startswith('/')
        else PROJECT_DIR / relative_path
    )

def get_data_path(relative_path: str) -> Path:
    return (
        Path(relative_path)
        if relative_path.startswith('/')
        else DATA_DIR / relative_path
    )