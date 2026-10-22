import logging

def get_root_logger(logger_name="mmseg", log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger
