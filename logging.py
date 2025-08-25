# src/utils/logging.py
import logging, os

def get_logger(name: str, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'), encoding='utf-8')
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger
