import argparse

import numpy as np

from src.environments.environment_loader import EnvironmentLoader
from src.utils.evaluations import EvaluationHandler
from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data


def get_perser_args():
    # Arguments for function
    parser = argparse.ArgumentParser(description='Test Agent in Production Scheduling Environment')

    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')
    parser.add_argument('-plot', '--plot-ganttchart', dest="plot_ganttchart", action="store_true",
                        help='Enable or disable model result plot.')
    parser.add_argument('-bf', '--binary_features', type=str, default="1001011000", required=False,
                        help='Binary list of features')
    parser.add_argument('-rh', '--run_heuristics', type=int, default=1, required=False,
                        help='Should run heuristics or not')

    args = parser.parse_args()

    return args


def main(external_config=None):

    # get config_file from terminal input
    parse_args = get_perser_args()
    config_file_path = parse_args.config_file_path

    # get config and data
    config = load_config(config_file_path, external_config)
    data = load_data(config)

    # Random seed for numpy as given by config
    np.random.seed(config['seed'])

    evaluation_handler = EvaluationHandler()

    for test_i in range(len(data)):
        # create env
        environment, _ = EnvironmentLoader.load(config, data=[data[test_i]])
        environment.runs = test_i


if __name__ == '__main__':
    main()
