"""
This file provides functions to parse and even generate scheduling problem instances based on BOM files.

Using this file requires a data_generation config. For example, it is necessary to specify
the type of the scheduling problem.
"""
# OS imports
import argparse
import os
import json
from datetime import datetime
import random
from pathlib import Path

# Config and data handling imports
from src.utils.file_handler.config_handler import ConfigHandler
from src.utils.file_handler.data_handler import DataHandler

# Functional imports
import copy
from typing import List
from src.data_generator.task import Task
from src.data_generator.sp_factory import SPFactory

def dfs_bom(node, sorted_top, tasks_mapping_ids, deadline, job_index, filename, quantity, should_multiply_quantity_to_execution_times, num_machines = 50,
            __machine_id ={}):
    """

    :param node:
    :param sorted_top:
    :param tasks_mapping_ids:
    :param deadline:
    :param job_index:
    :param filename:
    :param quantity:
    :param should_multiply_quantity_to_execution_times:
    :param num_machines:
    :param __machine_id: helper variable used to keep a link between instance_file machineID and machineID need by this implementation
    :return:
    """
    print("dfs_bom() parent quantity:", quantity, "curr operation_id", node['operationid'], "curr_op_quantity", node['quantity'], end=' cildren')
    for child in node.get('children', []):
        print(child['operationid'], end=",")
    print()
    for child in node.get('children', []):
        dfs_bom(child, sorted_top, tasks_mapping_ids, deadline - 1, job_index, filename, quantity * node['quantity'],
                should_multiply_quantity_to_execution_times, num_machines, __machine_id)

    machines = [0] * num_machines
    execution_times = {}
    setup_times = {}
    max_runtime = 0
    max_setup = 0
    average_runtime = 0
    for machine in node.get('machines', []):
        if machine['id'] not in __machine_id:
            __machine_id[machine['id']] = len(__machine_id)
        machine_id = __machine_id[machine['id']]
        #machine_id = machine['id'] - 1
        machines[machine_id ] = 1

        execution_times[machine_id] = machine['execution_time']
        max_runtime = machine['execution_time'] if max_runtime < machine['execution_time'] else max_runtime
        setup_times[machine_id] = machine['setup_time']
        max_setup =  machine['setup_time'] if max_setup < machine['setup_time'] else max_setup
        average_runtime +=  machine['execution_time']
    average_runtime = int(average_runtime / len(node.get('machines', [])))


    task = Task(job_index=job_index,
            task_index=len(sorted_top),
            task_id=node['operationid'],
            filename=filename,
            parent_index=node['parentid'],
            children=[tasks_mapping_ids[child['operationid']] for child in node.get('children', [])],
            quantity=node['quantity'] * quantity,
            machines=machines,
            execution_times=execution_times,
            setup_times=setup_times,
            deadline=deadline,
            runtime=max_runtime,
            average_runtime=average_runtime,
            setup_time=max_setup,
            tools=[],
            _n_tools=0,
            done=0,
            _n_machines=len(machines),
            should_multiply_quantity_to_execution_times=should_multiply_quantity_to_execution_times,
        )
    print("..dfs_bom() task quantity final:", task.quantity, "node quantity", node['quantity'], 'node id', task.task_id)
    sorted_top.append(task)
    sorted_top[-1].task_index = len(sorted_top) - 1
    tasks_mapping_ids[node['operationid']] = len(sorted_top) - 1


def get_job_deadline(start_date_str, delivery_date_str):
    # Convert strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S.%f")
    delivery_date = datetime.strptime(delivery_date_str, "%Y-%m-%d %H:%M:%S.%f")

    # Calculate the difference between the two dates as deadline
    deadline = (delivery_date - start_date).total_seconds()
    return deadline

def load_bom_files(input_directory, similar_instances_number, should_modify_instances, should_multiply_quantity_to_execution_times):
    instance_list: List[List[Task]] = []
    instance_name_list =[]

    # List all files in the directory
    files = os.listdir(input_directory)

    instance: List[Task] = []
    # Iterate through the files
    for file in files:
        # Check if the file is a regular file (not a directory)
        if file.endswith('.json') and os.path.isfile(os.path.join(input_directory, file)) :
            # Process the file
            with open(os.path.join(input_directory, file), 'r') as f:
                bom_job = json.load(f)
                deadline = get_job_deadline(bom_job['start_date'], bom_job['delivery_date'])
                num_machines = len(bom_job['metainfo']['machines_list'])
                __machine_id ={} #auxiliary variable used to map file machineId to 0, 1, ... as needed by this implementation
                tasks_mapping_ids = dict()
                sorted_top: List[Task] = []
                dfs_bom(bom_job, sorted_top, tasks_mapping_ids, deadline,
                        0, filename=file, quantity=1,
                        should_multiply_quantity_to_execution_times=should_multiply_quantity_to_execution_times,
                        num_machines=num_machines,
                        __machine_id = __machine_id
                )
                for task in sorted_top:
                    if task.parent_index is not None:
                        task.parent_index = tasks_mapping_ids[task.parent_index]

                #build: information number of nodes until root, CP (optimistic estimation)
                for task in sorted_top:
                    it_parent_index = task.parent_index
                    while it_parent_index is not None:
                        parent = sorted_top[it_parent_index]
                        task.no_remaining_operations += 1
                        task.remaining_work += min(parent.execution_times_setup.values())
                        it_parent_index = parent.parent_index

                instance_list.append(sorted_top)
                instance_name_list.append(Path(f.name).stem)

    original_list_length = len(instance_list)
    if should_modify_instances:
        for i in range(original_list_length):
            for task in instance_list[i]:
                 max_runtime = 0
                 max_setup = 0
                 average_runtime = 0
                 for machine_id in range(len(task.machines)):
                    machine_op_type = random.randint(0, 1)
                    print("load_bom_files", machine_op_type, "0- should add new machine if not existing, 1 - should modify current machine times")
                    # 0- should add new machine if not existing, 1 - should modify current machine times
                    if machine_op_type == 0:
                        if task.machines[machine_id] == 0:
                            task.machines[machine_id] = 1
                            task.execution_times[machine_id] = random.randint(10, task.runtime) * task.quantity
                            task.setup_times[machine_id] = random.randint(10, task.setup_time)
                            task.execution_times_setup[machine_id] = task.execution_times[machine_id] + task.setup_times[machine_id]
                    else:
                        task.execution_times[machine_id] = random.randint(10, task.runtime)
                        task.setup_times[machine_id] = random.randint(10, task.setup_time)
                        task.execution_times_setup[machine_id] = task.execution_times[machine_id] + task.setup_times[machine_id]
                    print("task",task.machines)
                    max_runtime = max(max_runtime, task.execution_times[machine_id])
                    max_setup = max(max_setup, task.setup_times[machine_id])
                    average_runtime += task.execution_times[machine_id]
                 task.runtime = max_runtime
                 task.average_runtime = int(average_runtime / len(task.machines))
                 task.setup_time = max_setup
                 task.recalculate_execution_times_setup()

        for i in range(similar_instances_number):
            instance_index = random.randint(0, original_list_length - 1)
            new_instance = copy.deepcopy(instance_list[instance_index])
            for task in new_instance:
                max_runtime = 0
                max_setup = 0
                average_runtime = 0
                for machine_id in range(len(task.machines)):
                   machine_op_type = random.randint(0, 1)
                   # 0- should add new machine if not existing, 1 - should modify current machine times
                   if machine_op_type == 0:
                       if task.machines[machine_id] == 0:
                           task.machines[machine_id] = 1
                           task.execution_times[machine_id] = random.randint(10, task.runtime)
                           task.setup_times[machine_id] = random.randint(10, task.setup_time)
                           task.execution_times_setup[machine_id] = task.execution_times[machine_id] + task.setup_times[machine_id]
                   else:
                        task.execution_times[machine_id] = random.randint(10, task.runtime)
                        task.setup_times[machine_id] = random.randint(10, task.setup_time)
                        task.execution_times_setup[machine_id] = task.execution_times[machine_id] + task.setup_times[machine_id]
                   max_runtime = max(max_runtime, task.execution_times[machine_id])
                   max_setup = max(max_setup, task.setup_times[machine_id])
                   average_runtime += task.execution_times[machine_id]
                task.runtime = max_runtime
                task.average_runtime = int(average_runtime / len(task.machines))
                task.setup_time = max_setup
                task.recalculate_execution_times_setup()

            instance_list.append(new_instance)
            instance_name_list.append(f'{instance_name_list[instance_index]}_similar_instances_{i}')

    return instance_list, instance_name_list

def main(config_file_path, external_config=None):

    # get config
    current_config: dict = ConfigHandler.get_config(config_file_path, external_config)

    # Create instance list
    instance_list, instance_name_list = load_bom_files(
        current_config.get('input_directory'),
        current_config.get('num_similar_instances'),
        current_config.get('should_modify_instances'),
        current_config.get('should_multiply_quantity_to_execution_times')
    )

    # comment if needed
    for instance in instance_list:
        for task in instance:
            print(task)

    # compute individual hash for each instance
    SPFactory.compute_and_set_hashes(instance_list)

    # Write resulting instance data to file
    data = {'instances' : instance_list, 'instances_names' :instance_name_list }
    if current_config.get('write_to_file', False):
        DataHandler.save_instances_data_file(current_config, data)


def get_parser_args():
    """Get arguments from command line."""
    # Arguments for function
    parser = argparse.ArgumentParser(description='Instance generation for scheduling optimization')
    parser.add_argument('-fp', '--config_file_path', type=str, required=True,
                        help='Path to config file you want to use for training')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parse_args = get_parser_args()
    config_file_path = parse_args.config_file_path
    main(config_file_path=config_file_path)
