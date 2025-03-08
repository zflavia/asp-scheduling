TODO: delete this
```bash
python3 -m src.agents.train_gnn -fp training/ppo/config_ASP_TUBES_ORIGINAL_TESTING_GNN.yaml
```




# Extended version of Schlably to support in-tree precedence relations

This is an extended version of Schlably to support in-tree precedence relations for ASP (Assembly Scheduling Problem) instances.
This version includes the following changes. More details about the original Schlably can be found below in this README.

Currently, only the indirect encoding was modified to support in-tree precedence relations. The direct encoding is not supported yet.

## Data Generation

Data processing and generation are adapted to parse and generate instances with in-tree precedence relations. The precedence relations are generated based on the in-tree structure of Bill-of-Material (BOM) file.
The BOM files were generated with the help of a more general ASP instance generator, which is not included in this repository available at [Scamp Datagen](https://github.com/acopie/scamp-datagen). There are different types of BOM files available in the data/own_data directory as examples.
Based on the given data generator config file for ASP, the bom_instance_factory.py script parses the BOM files into Task instances with in-tree precedence relations, and can also generate additional instances relying on (i) random perturbation of the processing times; (ii) random inclusion and exclusion of eligible machines. The perturbation keeps unaltered the number of operations as well as the minimal and maximal values of the processing times.

Example:

```bash
python -m src.data_generator.bom_instance_factory -fp data_generation/asp/config_ASP_TUBES_2.yaml
```
Modify the following fields in your config file for ASP case to generate new instances with in-tree precedence relations:

```yaml
# (O) [string] Load the bom input files from this input directory
input_directory: /Users/Documents/schlaby-asp/data/own_data/ASP_TUBES_2
# (0) [bool] If true, the instances will be modified
should_modify_instances: True
# (O) [int] Number of instances to be generated based on the original instances
num_similar_instances: 2
```

## Training

The training script is adapted to train the model with the observation space tailored for ASP instances with in-tree precedence relations. 

```bash
python -m src.agents.train -fp training/ppo/config_for_training.yaml 
```

In case you want to train the model with different features as mentioned in Section "Observation Space", you can set the command line argument `--binary_features` or `-bf` to the desired features.
The features are represented as a binary mask. The binary mask is a 10-bit binary number, where each bit represents a feature. The bit is set to 1 if the feature is included, and 0 otherwise.
The default binary mask is `1001011000`.

Example:

```bash
python -m src.agents.test -fp testing/ppo/config_for_testing.yaml -bf 1111100000
```

## Testing

The testing script is adapted to benchmark the model against the heuristics for ASP instances with in-tree precedence relations.

```bash
python -m src.agents.test -fp testing/ppo/config_for_testing.yaml 
```

In case you want to test the model and not benchmark it against the heuristics, you can set the command line argument `--run_heuristics` or `-rh` to `0` . 
The testing script will only test the model and save the results in the results directory. This is useful when you want to test the model with different features as mentioned in Section "Training and Testing for all combination of features" and the heuristics can be run only once.

```bash
python -m src.agents.test -fp testing/ppo/config_for_testing.yaml --run_heuristics 0
```

In case you want to test the model with different features as mentioned in Section "Observation Space", you can set the command line argument `--binary_features` or `-bf` to the desired features.
The features are represented as a binary mask. The binary mask is a 10-bit binary number, where each bit represents a feature. The bit is set to 1 if the feature is included, and 0 otherwise. 
The default binary mask is `1001011000`.

Example:

```bash
python -m src.agents.test -fp testing/ppo/config_for_testing.yaml -bf 1111100000
```

## Dispatching Rules - Heuristics

The dispatching rules  are heuristics that aim to select an operation from the list of feasible operations. The new and adapted rules for ASP (see heuristics_agent.py) are:
They can be configured in the config files. For example: `test_heuristics'
  ['LETSA', 'EDD_ASP', 'MPO_ASP', 'LPO_ASP', 'SPT_ASP', 'rand_asp']`:

- **Random Selection (RAND_ASP)**: At each step, a feasible operation is selected randomly. The rule name is 

- **Shortest Processing Times (SPT_ASP)**: Operations are selected in ascending order of their processing times. If there are several eligible machines for an operation, an estimation of the processing time is used (maximal or the average value over all machines).

- **Most Preceding Operations (MPO_ASP)**: The feasible operation with the largest number of predecessors is selected.

- **Least Preceding Operations (LPO_ASP)**: The feasible operation with the smallest number of predecessors is selected.

- **Most Remaining Operations (MRO_ASP)**: Operations are selected in decreasing order of the number of nodes in the ASP tree that belong to the path from the current node to the root. This rule is similar to the "maximum work remaining" rule used to select jobs in JSSP. A related rule was used in \cite{Zhang2018} in the case of flexible ASP.

- **Least Remaining Operations (LRO_ASP)**: Operations are selected in increasing order of the number of nodes in the ASP tree that belong to the path from the current node to the root.

- **Lead Time Evaluation and Scheduling Algorithm (LETSA)**: A heuristic rule designed for ASP that selects operations that belong to the current critical path (longest path, with respect to the execution times, from a feasible operation to the root of the ASP tree).


## Observation Space

The observation space in this project is inspired by previous works and consists of ten features tailored for the ASP. 

These features are:

1. **Operation Status (OS)**: Indicates the scheduling status of an operation. Index in the binary mask is 0.
   - `0`: Already scheduled.
   - `1`: Not scheduled but feasible.
   - `0.5`: Infeasible (at least one predecessor not scheduled).

2. **Estimated Processing Time per Operation (EPT)**: The weighted average of processing times for eligible machines for an operation. Index in the binary mask is 1.

3. **Estimated Completion Time per Operation (ECT)**: Index in the binary mask is 2.
   - If scheduled, it is the completion time.
   - If not scheduled, it is the maximum ECT of predecessors plus EPT.

4. **Estimated Remaining Processing Time per Operation (ERT_o)**: Sum of estimated processing times from the operation to the root of the operation network. Index in the binary mask is 3.

5. **Estimated Processing Time for the Next Operation (EPT_next)**: Estimated processing time for the successor of the current operation. Index in the binary mask is 4.

6. **Remaining Operations Count (ROC)**: Number of operations from the current node to the root, computed for unscheduled but feasible operations. Index in the binary mask is 5.

7. **Usage of the Critical Path (CP)**: Binary state indicating if the operation is on the critical path. Index in the binary mask is 6.

8. **Estimated Remaining Time per Machine (ERT_m)**: Sum of estimated processing times of unscheduled operations per machine. Index in the binary mask is 7.

9. **Assignment Matrix (AM)**: Binary matrix indicating eligible machines for each operation. Index in the binary mask is 8.

10. **Machine Sharing Feature (SF)**: Number of unscheduled operations per machine. Index in the binary mask is 9.

The feature index mapping can be found in the env_testris_scheduling.py file:

```python
self.feature_index_mapping = {
0: 'task_status',
1: 'operation_time_per_tasks',
2: 'completion_time_per_task',
3: 'estimated_remaining_processing_time_per_task',
4: 'estimated_remaining_processing_time_per_successor_task',
5: 'remaining_tasks_count',
6: 'is_task_in_critical_path',
7: 'remaining_processing_times_on_machines',
8: 'mat_machine_op',
9: 'machines_counter_dynamic'
}
```

The default binary mask is `1001011000`

## Training and Testing for all combination of features

The run.py script generates all combinations of features as binary masks for training and testing configurations and executes corresponding commands using the `subprocess` module. The masks are generated using the `product` function from the `itertools` module. The script is designed to be run from the command line and includes argument parsing for customization.

### Functions

1. **`execute_cmd(cmd)`**:
   - Executes a shell command.
   - Captures and prints the command's output.
   - Handles errors and prints error messages if the command fails.

2. **`generate_and_process_binary_masks(results_dir, train_config_file_path, test_config_file_path, lower_bound, upper_bound)`**:
   - Generates binary masks and processes them within the specified bounds.
   - Constructs and executes training and testing commands for each mask.
   - Skips the all-zero mask (`null_mask`).
   - Runs heuristics only once.

3. **`validate_bounds(lower_bound, upper_bound)`**:
   - Validates that `lower_bound` is greater than or equal to 1 and less than or equal to `upper_bound`.
   - Validates that `upper_bound` is less than or equal to 1023.
   - Raises an error if the bounds are invalid.

4. **`main()`**:
   - Parses command-line arguments.
   - Validates the bounds.
   - Calls `generate_and_process_binary_masks` with the parsed arguments.

### Command-Line Arguments

- `--results_dir`: Directory to store results (default: `'results/all/'`).
- `-trainfp`, `--train_config_file_path`: Path to the training config file (default: `'training/ppo/config_ASP_TUBES.yaml'`, required).
- `-testfp`, `--test_config_file_path`: Path to the testing config file (default: `'testing/ppo/config_ASP_TUBES_TESTING.yaml'`, required).
- `--upper_bound`: Upper bound for mask generation (default: `1023`, must be <= `1023`).
- `--lower_bound`: Lower bound for mask generation (default: `1`, must be >= `1`).

### Usage

Run the script from the command line with the desired arguments:

```sh
python run.py --results_dir 'results/custom/' -trainfp 'path/to/train_config.yaml' -testfp 'path/to/test_config.yaml' --upper_bound 1000 --lower_bound 10
```
e.g.  python run.py --results_dir results/asp_tubes_2 -trainfp training/ppo/config_ASP_TUBES_2.yaml -testfp testing/ppo/config_ASP_TUBES_TESTING_2.yaml --upper_bound 1023 --lower_bound 49



# Original Schlably
​
Schlably is a Python-Based framework for experiments on scheduling problems with Deep Reinforcement Learning (DRL). 
It features an extendable gym environment and DRL-Agents along with code for data generation, training and testing.  


Schlably was developed such that modules may be used as they are, but also may be customized to fit the needs of the user.
While the framework works out of the box and can be adjusted through config files, some changes are intentionally only possible through changes to the code.
We believe that this makes it easier to apply small changes without having to deal with complex multi-level inheritances.

Please see the [documentation](https://schlably.readthedocs.io/en/latest/index.html) for more detailed information and tutorials.

Check out the official [Preprint on arXiv](http://arxiv.org/abs/2301.04182)
​
## Install
Schlably, in its current version, only supports Python 3.10 and may be incompatible to other versions (e.g. 3.11).
To install all necessary Python packages run
   ````bash
   pip install -r requirements.txt
   ````
If you want to use [Weights&Biases](https://wandb.ai/site) (wandb) for logging, which we highly recommend,
you need to (create and) login with your account. Open a terminal and run:
   ````bash
   wandb login
   ````
  
​
## Quickstart

### Data Generation
To create your own data, or more precisely, instances of a scheduling problem, proceed as follows:
1. Create a custom data generation configuration from one of the configs listed in [config/data_generation/fjssp](config/data_generation/fjssp) or [config/data_generation/jssp](config/data_generation/jssp) (e.g. change number of machines, tasks, tools, runtimes etc.) to specify the generated instances.  
2. To generate the problem instances specified in your config, run 
   ````bash
   python -m src.data_generator.instance_factory -fp data_generation/jssp/<your_data_generation_config>.yaml
   ````
3. Please note that the file path needs to be given relative to the config directory of the project and that otherwise your config may not be found.

   python -m src.data_generator.instance_factory -fp data_generation/asp/config_job2_task50_tools0.yaml


python -m src.data_generator.bom_instance_factory -fp data_generation/asp/config_ASP_WIDE.yaml


​
### Training
To train your own model, proceed as follows:
1. To train a model, you need to specify a training config, such as the ones already included in [config/training](config/training). Note that different agents have different configs because the come with different algorithm parameters. You can customize the config to your needs, e.g. change the number of episodes, the learning rate, the batch size etc.
    - We are using weights & biases (wandb) to track our results.
   If you want to track your results online, create your project at wandb.ai and set config parameter wandb_mode to 1 (offline tracking) or 2 (online tracking)
   and specify *wandb_project* parameter in config file and *wandb_entity* constant in the [logger.py](src/utils/logger.py)
2. Run 
   ````bash
   python -m src.agents.train -fp training/ppo/<your_training_config>.yaml
   ````
3. Please note that the file path needs to be given relative to the config directory of the project and that otherwise your config may not be found.

Immediately after training the model will be tested and benchmarked against all heuristics included in the TEST_HEURISTICS constant located in [src/agents/test.py](src/agents/test.py)
The trained model can be accessed via the experiment_save_path and saved_model_name you specified in the training config.

python -m src.agents.train -fp training/ppo/config_ASP_WIDE.yaml

​
### Testing
As aforementioned, [train.py](src/agents/train.py) automatically tests the model once training is complete. If you want to test a certain model again, or benchmark it against other heuristics, proceed as follows:
1. As in training, you need to point to a testing config file like the ones provided in [config/testing](config/testing).  You may change entries according to your needs.
We provide a pre-trained masked PPO model. Thus, creating a config in [config/testing/ppo_masked](config/testing/ppo_masked) and assigning *example_ppo_masked_agent* to *saved_model_name* allows you to test without training first.
2. Run 
   ````bash
   python -m src.agents.test -fp testing/ppo_masked/<your_testing_config>.yaml
   ````
3. Please note that the file path needs to be given relative to the config directory of the project and that otherwise your config may not be found.
4. Optionally, you may use the parameter --plot-ganttchart to plot the test results.

We have pre-implemented many common priority dispatching rules, such as Shortest Processing Time first, and a flexible optimal solver.

​
## Advanced config handling
On the one hand running data_generation, training or testing by specifying a config file path offers an easy and comfortable way to use this framework, but on the other hand it might seem a bit restrictive. 
Therefore, there is the possibility to start all three files by passing a config dictionary to their main functions. 
This comes in handy, if you need to loop across multiple configs or if you want to change single parameters without saving new config files.


### Training via external config
1. Import the main function from agents.train to your script 
2. Load a config or specify an entire config
3. Loop over the parameter you want to change and update your config before starting a training 
   ```` python
   # import training main function
   from agents.train import main as training_main

   # load a default config
   train_default_config = ConfigHandler.get_config(DEFAULT_TRAINING_FILE)
   # specify your seeds
   seeds = [1455, 2327, 7776]
   
   # loop
   for seed in seeds:
      train_default_config.update({'seed': seed})
      
      # start training
      training_main(external_config=train_default_config)
   ````

  
## wandb Sweep
If you want to use wandb for hyperparameter sweeps, simply follow these instructions
1. Edit [config/sweep/config_sweep_ppo.yaml](config/sweep/config_sweep_ppo.yaml) or create an own sweep config. 
   1. [Here](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb) you can find instructions for creating a configuration.
   2. Make sure that you point to the right training config in your sweep config file. 
   3. In the training config which you are using as the basis for the sweep, make sure to track the right success metric. You find it under "benchmarking" along with instructions.
2. Run ``wandb sweep -e your_entity -p your_project config/sweep/config_sweep_ppo.yaml`` to create a wandb sweep.
3. Run ``wandb agent your_entity/your_project/sweep_name`` to start the sweep

## Structure
For everyone who wants to work with the project beyond training and testing within the current possibilities, important files and functions are explained in the following section. 
​
### [data_generator/task.py](src/data_generator/task.py)
1. Class Task
    - Task objects are used to represent tasks and their dependencies in a real world production process
    - Tasks with same job_num can be viewed as belonging to the same job
    - A list of task objects builds an instance, which can be put into the environment
​
### [data_generator/instance_factory.py](src/data_generator/instance_factory.py)
- Can be executed to generate new instances for a scheduling problem as data for training or testing
- Automatically computes deadlines for tasks according to random heuristic. You can change the heuristic used for computation by adapting *DEADLINE_HEURISTIC* constant 

### [data_generator/sp_factory.py](src/data_generator/sp_factory.py)
1. Function *generate_instances*
   - This function will be called when generating instances
   - If you want to modify the task generation beyond what is possible by customizing the data_generation config, you can do it here (e.g. different random functions)

​
### [env_tetris_scheduling.py](src/environments/env_tetris_scheduling.py)
- This environment tracks the current scheduling state of all tasks, machines and tools and adjusts the state according to valid incoming actions 
- Action and state space are initialized, depending on the number of tasks, machines and tools 
​
1. Function *Step*
    - Selected actions (select a job) are processed here
    - Without action masking, actions are checked on validity
    - Returning reward and the current production state as observation
    
2. Function *compute_reward*
    - Returns dense reward for makespan optimization according to [https://arxiv.org/pdf/2010.12367.pdf](https://arxiv.org/pdf/2010.12367.pdf)
    - You can set custom reward strategies (e.g. depending on tardiness or other production scheduling attributes provided by the env)
​
3. Function *state_obs* 
    - For each task, scales runtime, task_index, deadline between 0-1
    - This observation will be returned from the environment to the agent
    - You can set other production scheduling attributes as return or different normalization 
    
​
### [agents/heuristic/heuristic_agent.py](src/agents/heuristic/heuristic_agent.py)
- This file contains following heuristics, which can be used to choose actions based on the current state
  - Random task
  - EDD: earliest due date
  - SPT: shortest processing time first
  - MTR: most tasks remaining
  - LTR: least tasks remaining
- You can add own heuristics as function to this file
1. Class *HeuristicSelectionAgent*
   - *__call__* function selects the next task action according to heuristic passed as string
   - If you added own heuristics as functions to this file, you have to add your_heuristic_function_name: "your_heuristic_string" item to the *task_selections* dict attribute of this class

​
### [agents/intermediate_test.py](src/agents/intermediate_test.py)
- During training, every *n_test_steps* (adjust in train.py *IntermediateTest*)  *_on_step* tests and saves the latest model (save only if better results than current saved model)

​
### [utils/evaluation.py](src/utils/evaluations.py)
- Function *evaluate_test* gets all test results, then calculate and returns evaluated parameters
- You can add or remove metrics here

​
### [utils/logger.py](src/utils/logger.py)
1. Class *Logger*
   - *record* can be called to store a dict with parameters in a logger object
   - calling *dump* tracks the stored parameters and clears the logger object memory buffer. 
   - At the moment only wandb logging is supported. Own logging tools (e.g. Tensorboard) can be implemented und then used by calling them in the *dump* function
