import logging

logging.basicConfig(
    level=logging.INFO
)


def get_logger(module_name):
    return logging.getLogger(name=module_name)
