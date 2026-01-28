"""
This file provides functions to train an agent on a scheduling-problem environment.
By default, the trained model will be evaluated on the test data after training,
by running the test_model_and_heuristic function from test.py.

Using this file requires a training config. For example, you have to specify the algorithm used for the training.

There are several constants, which you can change to adapt the training process:
"""
# OS imports
import argparse
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from copy import deepcopy
import yaml

# Functional imports
from src.agents import intermediate_test
from src.agents import test
from src.environments.environment_loader import EnvironmentLoader
from src.utils.file_handler.data_handler import DATA_DIRECTORY
from src.utils.file_handler.model_handler import ModelHandler
from src.data_generator.task import Task
from src.utils.logger import Logger
from src.agents.train_test_utility_functions import get_agent_class_from_config, load_config, load_data
from datetime import datetime


def final_evaluation(config: dict, data_test: List[List[Task]], logger: Logger):
    """
    Evaluates the trained model and logs the gp

    :param config: Training config
    :param data_test: Dataset with instances to be used for the test
    :param logger: Logger object

    :return: None

    """

    # Create wandb artifact from local file to store best model with wandb
    best_model_path = ModelHandler.get_best_model_path(config)
    logger.log_wandb_artifact({'name': 'agent_model', 'type': 'model'}, file_path=best_model_path.with_suffix('.pkl'))

    # test model
    # create env and agent
    agent = get_agent_class_from_config(config)
    best_model = agent.load(file=best_model_path, config=config, logger=logger)
    evaluation_results = test.test_model_and_heuristic(config=config, model=best_model, data_test=data_test,
                                                       logger=logger, plot_ganttchart=False, log_episode=True)

    # log the metric which you find most relevant (this should be used to optimize a hyperparameter sweep)
    success_metric = evaluation_results['agent'][config.get('success_metric')]
    logger.record({'success_metric': success_metric})
    logger.dump()

    # log evaluation to wandb
    logger.write_to_wandb_summary(evaluation_results)


def training(config: dict, data_train: List[List[Task]], data_val: List[List[Task]], logger: Logger, binary_features = None) -> None:
    """
    Handles the actual training process.
    Including creating the environment, agent and intermediate_test object. Then the agent learning process is started

    :param config: Training config
    :param data_train: Dataset with instances to be used for the training
    :param data_val: Dataset with instances to be used for the evaluation
    :param logger: Logger object used for the whole training process, including evaluation and testing

    :return: None

    """
    # create Environment
    #TODO: Roxana way 'binary_features' 2 times env, _ = EnvironmentLoader.load(config, binary_features, data=data_train, binary_features=binary_features)
    env, _ = EnvironmentLoader.load(config, data=data_train, binary_features=binary_features)

    # create Agent model
    agent = get_agent_class_from_config(config)(env=env, config=config, logger=logger)

    # create IntermediateTest class to save new optimum model every <n_test_steps> steps
    inter_test = intermediate_test.IntermediateTest(env_config=config,
                                                    n_test_steps=config.get('intermediate_test_interval'),
                                                    data=data_val, logger=logger, binary_features=binary_features)

    # Actual "learning" or "training" phase
    return agent.learn(total_instances=config['total_instances'], total_timesteps=config['total_timesteps'],
                intermediate_test=inter_test)


def run_training(config_file_name: dict = None, external_config: dict = None, binary_features = None) -> None:
    """
    Main function to train an agent in a scheduling-problem environment.

    :param config_file_name: path to the training config you want to use for training
        (relative path from config/ folder)
    :param external_config: dictionary that can be passed to overwrite the config file elements
    :param binary_features:

    :return: None
    """

    # get config and data
    config = load_config(config_file_name, external_config)
    stored_instances = load_data(config)

    data = stored_instances['instances']
    data_names = stored_instances['instances_names']

    # create logger and update config
    logger = Logger(config=config)
    config = logger.config

    #to run with multiple seeds
    seeds = config['seed']
    if isinstance(seeds, int):
        seeds = [seeds]
    else:
        seeds = seeds.copy()

    model_name =  config['saved_model_name']

    makespans = 0
    for seed in seeds:
        # Random seed for numpy as given by config
        np.random.seed(seed)
        config['seed'] = seed
        config['saved_model_name'] = f'{model_name}_seed_{seed}'

        # train/test/validation data split
        split_random_seed = seed if not config.get('overwrite_split_seed', False) else 1111
        print("dataset len:",len(data))
        train_data, test_data = train_test_split(
            data, train_size=config.get('train_test_split'), random_state=split_random_seed)
        test_data, val_data = train_test_split(
            test_data, test_size=config.get('test_validation_split'), random_state=split_random_seed)

        # log data
        logger.log_wandb_artifact({'name': 'dataset', 'type': 'dataset',
                                   'description': 'job_config dataset, split into test, train and validation',
                                   'metadata': {'train_test_split': config.get('train_test_split'),
                                                'test_validation_split': config.get('test_validation_split')}},
                                  file_path=DATA_DIRECTORY / config['instances_file']
                                  )
        # training
        makespans += training(config=config, data_train=train_data, data_val=val_data, logger=logger, binary_features=binary_features)
    return makespans / len(seeds)

def get_parser_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description='Train Agent')
    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')
    parser.add_argument('-bf', '--binary_features', type=str, default='1001011000', required=False,
                        help='Binary list of features')

    # optuna flags
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="gp_dr_tuning")
    parser.add_argument("--storage", type=str, default=None)  # ex: sqlite:///optuna.db
    parser.add_argument("--seed", type=int, default=42)  # sampler seed
    #parser.add_argument("--direction", type=str, default="minimize")
    parser.add_argument("--seed-mode", type=str, default="mean_over_seeds",
                        choices=["mean_over_seeds", "pick_one_seed"])

    return parser.parse_args()

def normalize_seeds(cfg: dict) -> list[int]:
    s = cfg.get("seed", 0)
    return s if isinstance(s, list) else [s]


def build_objective(base_cfg: dict, mode: str = "mean_over_seeds"):
    """
    mode:
      - "pick_one_seed": Optuna alege un seed din lista din YAML per trial
      - "mean_over_seeds": rulezi pe toate seed-urile și întorci media (recomandat)
    """
    base_seeds = normalize_seeds(base_cfg)

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)

        # ---- Search space (adaptează dacă vrei range-uri mai mari) ----
        cfg["gp_population_size"] = trial.suggest_int("population_size", 50, 400, step=50)
        cfg["gp_tree_max_depth"] = trial.suggest_int("gp_tree_max_depth", 2, 10)
        cfg["gp_tree_initial_max_depth"] = trial.suggest_int("gp_tree_initial_max_depth", 2, 5)
        #cfg["gp_population_variation"] = trial.suggest_float("gp_population_variation", 0.5, 0.95)
        cfg["gp_generations_number"] = trial.suggest_int("generations_number", 10, 200, step=10)
        cfg["gp_simplify_frequency"] = trial.suggest_int("simplify_frequency", 5, 50, step=5)
        #cfg["gp_tournament_size"] = trial.suggest_int("gp_tournament_size", 5, 50, step=5)
        cfg["gp_aos_type"] = trial.suggest_categorical("gp_aos_type", ["aos", "epsilon-qlearning", "random"])

        # cfg["gp_population_size"] = trial.suggest_int("population_size", 10, 40, step=10)
        # cfg["gp_tree_max_depth"] = trial.suggest_int("gp_tree_max_depth", 2, 8)
        # cfg["gp_tree_initial_max_depth"] = trial.suggest_int("gp_tree_initial_max_depth", 2, 5)
        # cfg["gp_population_variation"] = trial.suggest_float("gp_population_variation", 0.85, 0.95)
        # cfg["gp_generations_number"] = trial.suggest_int("generations_number", 50, 150, step=50)
        # cfg["gp_simplify_frequency"] = trial.suggest_int("simplify_frequency", 10, 50, step=10)
        # cfg["gp_tournament_size"] = trial.suggest_int("simplify_frequency", 2, 6, step=2)
        # cfg["gp_aos_type"] = trial.suggest_categorical("gp_aos_type", ["aos", "epsilon-qlearning", "random"])

        # ---- Seed handling ----
        if mode == "pick_one_seed":
            cfg["seed"] = trial.suggest_categorical("seed", base_seeds)
            return run_training(config_file_name=None, external_config=cfg, binary_features=None)
        # recommended: reduce noise by averaging over seeds
        scores = []
        for s in base_seeds:
            cfg_one = deepcopy(cfg)
            cfg_one["seed"] = s
            scores.append(run_training(config_file_name=None, external_config=cfg, binary_features=None))

        return sum(scores) / len(scores)

    return objective

if __name__ == "__main__":

    args = get_parser_args()
    config_file_path = args.config_file_path
    binary_features = args.binary_features

    if not args.optuna:
        start_time = datetime.now()
        run_training(config_file_name=config_file_path, binary_features=binary_features)
        end_time = datetime.now()
        timespan = (end_time - start_time).total_seconds()  # * 1000
        print(f"Training timestamp: {timespan} seconds")

    else:
        print('Start optuna study')

        sampler = optuna.samplers.TPESampler(seed=args.seed)

        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=sampler,
            storage=args.storage,
            load_if_exists=True,
        )

        base_cfg = load_config(config_file_path, external_config=None)
        objective = build_objective(base_cfg, mode=args.seed_mode)
        study.optimize(objective, n_trials=args.n_trials)

        print("Best value:", study.best_value)
        print("Best params:", study.best_params)
