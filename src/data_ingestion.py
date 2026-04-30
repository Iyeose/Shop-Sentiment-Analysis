import pandas as pd

from pathlib import Path
import re
import numpy as np
import logging
from config.constant import Input_Data

logging.basicConfig(level=logging.INFO)

def data_ingestion():
    try:
        data = pd.read_csv(Input_Data)
        logging.info("Data successfully loaded")
        print(data.head(5))
        return data
    except Exception as e:
        logging.error(f"error occurred while loading the data {e}")

data_ingestion()