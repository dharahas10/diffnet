import logging
import logging.config
import os
import sys
from datetime import datetime

from util.data_module import DataModule

LOG_DIR = "./logs"


def setup_logging():
    """Load logging configuration"""
    config_path = "./config/logging.ini"

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = datetime.now().strftime("%y%m%d-%H:%M:%S")
    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": f"{LOG_DIR}/{timestamp}.log"},
    )


if __name__ == "__main__":

    # setup logging
    setup_logging()
    log = logging.getLogger(__name__)

    log.info("Diffnet model training started")

    data_dir = "./data/yelp"
    if not os.path.isdir(data_dir):
        log.error("Data directory not found")
        sys.exit()

    # load data
    log.info("Loading dataset")

    data_module = DataModule(data_dir)
    data_module.load()
