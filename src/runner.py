import yaml
from yaml.loader import SafeLoader

from logger import get_logger


# pd.set_option("display.max_colwidth", 100)
logger = get_logger(module_name="data-cleaning")


def parse_config_file(file_path: str):
    with open(file_path, "r") as f:
        config_data = list(yaml.load_all(f, Loader=SafeLoader))
        logger.info(config_data)


def run_app(config_file_path: str):
    parse_config_file(file_path=config_file_path)
