import argparse
import pandas as pd

def get_argument_parser_basic(desc):
    pd.set_option('display.max_rows', 37)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', dest='data_directory', type=str, help="Data directory root.", required=True)
    return parser

def get_argument_parser_uncore(desc):
    parser = get_argument_parser_basic(desc)
    parser.add_argument('--uncore', dest='uncore', type=str, help="What uncore counters to include (default 'shared').",
                        default='shared', choices=['all', 'shared', 'exclusive', 'none'])
    return parser

def get_argument_parser(desc):
    parser = get_argument_parser_uncore(desc)
    parser.add_argument('--cutoff', dest='cutoff', type=float, default=1.25, help="Cut-off for labelling the runs.")
    parser.add_argument('--config', dest='config', nargs='+', type=str, help="Which configs to include (L3-SMT, L3-SMT-cores, ...).", default=['L3-SMT'])
    parser.add_argument('--alone', dest='include_alone', action='store_true', default=False, help="Include alone runs.")

    return parser