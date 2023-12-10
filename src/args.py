from argparse import ArgumentParser
from typing import Dict

"""
In this module we are taking arguments from command line while running the project

We have only one argument as of now - 
    python3 src/main.py -c "path_to_config_file"
    
Here config file is a .yaml file
    
"""


def parse_args(input_args) -> Dict[str, str]:
    parser = ArgumentParser()

    parser.add_argument("-c", "--config-file", type=str, required=True)

    args = vars(parser.parse_args(input_args))

    return args
