import logging

logging.basicConfig(
    level=logging.INFO
)

"""
This is for logging purposes to monitor the flow of the code
For now the logging level used is upto INFO only
"""


def get_logger(module_name):
    return logging.getLogger(name=module_name)
