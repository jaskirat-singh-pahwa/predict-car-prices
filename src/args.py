from argparse import ArgumentParser
from typing import Dict


def parse_args(input_args) -> Dict[str, str]:
    parser = ArgumentParser()

    parser.add_argument("-c", "--config-file", type=str, required=True)

    args = vars(parser.parse_args(input_args))

    return args
