import subprocess
from itertools import product
import argparse

def execute_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")


def generate_and_process_binary_masks(results_dir, train_config_file_path, test_config_file_path, lower_bound, upper_bound):
    prefix_train_cmd = f"python -m src.agents.train -fp {train_config_file_path} -bf"
    prefix_test_cmd = f"python -m src.agents.test -fp {test_config_file_path} -bf"

    null_mask = '0000000000'

    counter = 0
    run_heuristics = 1 # should run the heuristics and print their gp only once if =1
    for mask_tuple in product([0, 1], repeat=10):
        mask = ''.join(map(str, mask_tuple))
        if mask != null_mask:
            if counter >= lower_bound and counter <= upper_bound:
                print(mask, counter)
                train_cmd = f"{prefix_train_cmd} {mask} > {results_dir}train_{counter}_{mask}.txt"
                test_cmd = f"{prefix_test_cmd} {mask} -rh {run_heuristics} > {results_dir}test_{counter}_{mask}.txt"

                execute_cmd(train_cmd)
                execute_cmd(test_cmd)
                run_heuristics = 0
            elif counter > upper_bound:
                return
        counter += 1

def validate_bounds(lower_bound, upper_bound):
    if lower_bound < 1 or lower_bound > upper_bound:
        raise argparse.ArgumentTypeError("lower_bound must be >= 1 and <= upper_bound")
    if upper_bound > 1023:
        raise argparse.ArgumentTypeError("upper_bound must be <= 1023")
    return lower_bound, upper_bound

def main():
    parser = argparse.ArgumentParser(description='Generate and process binary masks for training and testing')

    parser.add_argument('--results_dir', type=str, default='gp/all/', help='Directory to store gp')
    parser.add_argument('-trainfp', '--train_config_file_path', default='training/ppo/config_ASP_TUBES.yaml', required=True,
                        help='Path to config file you want to use for training')
    parser.add_argument('-testfp', '--test_config_file_path', default='testing/ppo/config_ASP_TUBES_TESTING.yaml', required=True,
                        help='Path to config file you want to use for testing')
    parser.add_argument('--upper_bound', type=int, default=1023, help='Upper bound for mask generation (must be <= 1023)')
    parser.add_argument('--lower_bound', type=int, default=1, help='Lower bound for mask generation (must be >= 1)')

    args = parser.parse_args()
    train_config_file_path = args.train_config_file_path
    test_config_file_path = args.test_config_file_path
    results_dir = args.results_dir
    lower_bound, upper_bound = validate_bounds(args.lower_bound, args.upper_bound)

    generate_and_process_binary_masks(results_dir, train_config_file_path, test_config_file_path, lower_bound, upper_bound)

if __name__ == '__main__':
    main()


