"""
This module provides the following scheduling heuristics as function:

- EDD: earliest due date
- SPT: shortest processing time first
- MTR: most tasks remaining
- LTR: least tasks remaining
- Random: random action

You can implement additional heuristics in this file by specifying a function that takes a list of tasks and an action
mask and returns the index of the job to be scheduled next.

If you want to call your heuristic via the HeuristicSelectionAgent or edit an existing shortcut,
adapt/extend the task_selection dict attribute of the HeuristicSelectionAgent class.

:Example:

Add a heuristic that returns zeros (this is not a practical example!)
1. Define the according function

.. code-block:: python

    def return_0_heuristic(tasks: List[Task], action_mask: np.array) -> int:
        return 0

2. Add the function to the task_selection dict within the HeuristicSelectionAgent class:

.. code-block:: python

    self.task_selections = {
        'rand': random_task,
        'EDD': edd,
        'SPT': spt,
        'MTR': mtr,
        'LTR': ltr,
        'ZERO': return_0_heuristic
    }

"""
import numpy as np
from typing import List
import random
import copy


from src.data_generator.task import Task

def get_active_task_dict(tasks: List[Task]) -> dict:
    """
    Helper function to determining the next unfinished task to be processed for each job

    :param tasks: List of task objects, so one instance

    :return: Dictionary containing the next tasks to be processed for each job

    Would be an empty dictionary if all tasks were completed

    """
    active_job_task_dict = {}
    for task_i, task in enumerate(tasks):
        if not task.done and task.job_index not in active_job_task_dict.keys():
            active_job_task_dict[task.job_index] = task_i

    return active_job_task_dict

def get_active_task_dict_asp(tasks: List[Task]) -> dict:
    """
    Helper function to determining the next unfinished task to be processed for each job

    :param tasks: List of task objects, so one instance

    :return: Dictionary containing the next tasks to be processed for each job

    Would be an empty dictionary if all tasks were completed

    """
    active_task_dict = {}
    for task_i, task in enumerate(tasks):
        are_children_done = True
        for task_j, sub_task in enumerate(tasks):
            if sub_task.task_index in task.children:
                if not sub_task.done:
                    are_children_done = False

        if not task.done and are_children_done is True and task.task_index not in active_task_dict.keys():
            active_task_dict[task.task_index] = 1

    return active_task_dict

def is_leaf(task: Task):
    return len(task.children) == 0 and task.parent_index

critical_path = ([], 0)

def compute_paths(tasks: List[Task], task: Task, path, duration, visited):
    global critical_path
    path.append(task.task_index)

    # Compute the length (cumulative processing # time) of each path determined in step 4.1.
    duration += task.max_execution_times_setup

    # 4.3 a: Determine the critical (the largest cumulative processing time) path
    if is_leaf(task) and duration > critical_path[1]:
        critical_path = (copy.deepcopy(path), duration)
        return

    for _, index_subtask in enumerate(task.children):
        compute_paths(tasks, tasks[index_subtask], path, duration, visited)
        path.pop()

def letsa(tasks: List[Task], action_mask: np.array, feasible_tasks, visited, max_deadline):
    global critical_path
    critical_path = ([], 0)
    length = len(feasible_tasks)
    print(length)
    max_path_length_task_index = feasible_tasks[0]
    max_path_length_delete_index = 0
    compute_paths(tasks, tasks[max_path_length_task_index], [], 0, visited)
    max_length_critical_path = (critical_path[0].copy(), critical_path[1])
    
    for i in range(len(feasible_tasks)):
        critical_path = ([], 0)
        compute_paths(tasks, tasks[feasible_tasks[i]], [], 0, visited)
        if max_length_critical_path[1] < critical_path[1]: 
            max_path_length_task_index  = feasible_tasks[i]
            max_path_length_delete_index = i
            max_length_critical_path = (critical_path[0].copy(), critical_path[1])
   
    print('Length of critical path', max_length_critical_path[1])
    for i in range(len(max_length_critical_path[0])):
        print('Task_index:', max_length_critical_path[0][i], 'Task_id in BOM: ', tasks[max_length_critical_path[0][i]].task_id, ' Quantity: ', tasks[max_length_critical_path[0][i]].quantity, ' Runtime: ', tasks[max_length_critical_path[0][i]].max_execution_times_setup)

    # 4.3 b Select the operation Je of the critical path that also belongs to the feasible list F.
    # in this case it is the first operation, which also belongs to F, that is selected for scheduling.

    # 4.4 Set its tentative completion time Ce equal to: (i) the starting time of operation Je from the
    # partial schedule (constraint 2.2 of (PI)), (ii) the due-date De if operation c is the last
    # operation of the final assembly Pe (constraint 2.4 of (PI).
    completion_time = 0
    if not tasks[max_path_length_task_index].parent_index:
        completion_time = max_deadline
    else:
        # ??? Choose the earliest starting time of all successors Jc
        completion_time = max_deadline
        parent_start_task_index = tasks[max_path_length_task_index].parent_index
        if parent_start_task_index and tasks[parent_start_task_index].done and tasks[parent_start_task_index].started < completion_time:
            completion_time = tasks[parent_start_task_index].started # - EPSILON
        # else:
        #     print('Else branch', parent_start_task_index, tasks[parent_start_task_index].done, tasks[parent_start_task_index].started, completion_time)
        # if tasks[parent_start_task_index].started > completion_time:
        #     print('parent_start_task_index', parent_start_task_index, 'Completion time:', completion_time, 'Start time:', tasks[parent_start_task_index].started)


    # 4.5 Compute the starting time based on the available machines that can produce the operation Jc
    # 4.6 Schedule operation Jc at the latest available starting time Sc on the corresponding machine

    # #  NOTE: instead of scheduling right away, skip it now, follow the next steps and only at the end return the index of the start of operation to be scheduled
    # # 4.7 Delete operation Jc from the operation network.
    # tasks[start_task_index].deleted = True
    # # 4.8 Add all operations Ji such that di = Jc, to the feasible list.
    # # Also check is the operation was not added in the list
    # # Priority is given to the predecessor in the critical path while updating the list of feasible operations
    del feasible_tasks[max_path_length_delete_index]
    for _, sub_task_index in enumerate(tasks[max_path_length_task_index].children):
        if not tasks[sub_task_index].done:
            feasible_tasks.append(sub_task_index)
    print('Task: ', tasks[max_path_length_task_index].task_id, ' completion_time: ', completion_time)
    print('FEASIBLE TASKS: ')
    for i in range(len(feasible_tasks)):
        print(tasks[feasible_tasks[i]].task_id, tasks[tasks[feasible_tasks[i]].parent_index].task_id, tasks[tasks[feasible_tasks[i]].parent_index].started)
    return max_path_length_task_index, int(completion_time)


def edd_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    EDD: earliest due date. Determines the task with the smallest deadline

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = -1
    earliest_due_date = np.inf
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys() and task.deadline < earliest_due_date and not task.done:
            earliest_due_date = task.deadline
            task_index = i
    return task_index


def mpo_root_asp(tasks: List[Task], action_mask: np.array) -> int:
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    max_number_children = -1
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            remaining_tasks_count = 0
            task_successor_index = task.parent_index
            while task_successor_index is not None:
                remaining_tasks_count += 1
                task_successor_index = tasks[task_successor_index].parent_index
            if max_number_children < remaining_tasks_count:
                max_number_children = remaining_tasks_count
                task_index = task.task_index
    return task_index


def mpo_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    MPO (Maximal Predecessor Operations Rule): Determines the task with the highest number of predecessors.

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    max_number_children = 0
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            if len(task.children) >= max_number_children:
                max_number_children = len(task.children)
                task_index = task.task_index
    return task_index


def lpo_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    LPO (Least Predecessor Operations Rule): Determines the task with the least number of predecessors.

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """

    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    min_number_children = np.inf
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            if len(task.children) < min_number_children:
                min_number_children = len(task.children)
                task_index = task.task_index
    return task_index


def lpo_root_asp(tasks: List[Task], action_mask: np.array) -> int:
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = 0
    min_number_children = np.inf
    for i, task in enumerate(tasks):
        if task.task_index in possible_tasks.keys():
            remaining_tasks_count = 0
            task_successor_index = task.parent_index
            while task_successor_index is not None:
                remaining_tasks_count += 1
                task_successor_index = tasks[task_successor_index].parent_index
            if min_number_children > remaining_tasks_count:
                min_number_children = remaining_tasks_count
                task_index = task.task_index
    return task_index

def spt_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    SPT: shortest processing time first. Determines the unfinished task has the lowest runtime for ASP

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the task selected according to the heuristic

    """
    possible_tasks = get_active_task_dict_asp(tasks)
    task_index = -1
    shortest_processing_time = np.inf
    for i, task in enumerate(tasks):
        if not task.done and task.task_index in possible_tasks.keys():
            if task.runtime < shortest_processing_time:
                shortest_processing_time = task.runtime
                task_index = i
    return task_index

def random_task_asp(tasks: List[Task], action_mask: np.array) -> int:
    """
    Returns a random task

    :param tasks: Not needed
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    possible_tasks = get_active_task_dict_asp(tasks)
    random_task = random.choice(list(possible_tasks.keys()))
    return random_task

def edd(tasks: List[Task], action_mask: np.array) -> int:
    """
    EDD: earliest due date. Determines the job with the smallest deadline

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        num_jobs = action_mask.shape[0] - 1
        num_tasks_per_job = len(tasks) / num_jobs
        deadlines = np.full(num_jobs + 1, np.inf)

        for job_i in range(num_jobs):
            idx = int(num_tasks_per_job * job_i)
            deadlines[job_i] = tasks[idx].deadline

        deadlines = np.where(action_mask == 1, deadlines, np.full(deadlines.shape, np.inf))
        chosen_job = np.argmin(deadlines)
    return chosen_job

def spt(tasks: List[Task], action_mask: np.array) -> int:
    """
    SPT: shortest processing time first. Determines the job of which the next unfinished task has the lowest runtime

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        num_jobs = action_mask.shape[0] - 1
        runtimes = np.full(num_jobs + 1, np.inf)
        active_task_dict = get_active_task_dict(tasks)

        for i in range(num_jobs):
            if i in active_task_dict.keys():
                task_idx = active_task_dict[i]
                runtimes[i] = tasks[task_idx].runtime
        runtimes = np.where(action_mask == 1, runtimes, np.full(runtimes.shape, np.inf))
        chosen_job = np.argmin(runtimes)
    return chosen_job


def mtr(tasks: List[Task], action_mask: np.array) -> int:
    """
    MTR: most tasks remaining. Determines the job with the least completed tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        tasks_done = np.zeros(len(tasks) + 1)
        possible_tasks = get_active_task_dict(tasks)
        for _, task in enumerate(tasks):
            if task.done and task.job_index in possible_tasks.keys():
                tasks_done[possible_tasks[task.job_index]] += 1

        task_mask = np.zeros(len(tasks) + 1)
        for job_id, task_id in possible_tasks.items():
            if action_mask[job_id] == 1:
                task_mask[task_id] += 1
        tasks_done = np.where(task_mask == 1, tasks_done, np.full(tasks_done.shape, np.inf))
        tasks_done[-1] = np.inf
        chosen_task = np.argmin(tasks_done)
        chosen_job = tasks[chosen_task].job_index
    return chosen_job


def ltr(tasks: List[Task], action_mask: np.array) -> int:
    """
    LTR: least tasks remaining. Determines the job with the most completed tasks

    :param tasks: List of task objects, so one instance
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        tasks_done = np.zeros(len(tasks) + 1)
        possible_tasks = get_active_task_dict(tasks)
        for _, task in enumerate(tasks):
            if task.done and task.job_index in possible_tasks.keys():
                tasks_done[possible_tasks[task.job_index]] += 1
        task_mask = np.zeros(len(tasks) + 1)
        for job_id, task_id in possible_tasks.items():
            if action_mask[job_id] == 1:
                task_mask[task_id] += 1
        tasks_done = np.where(task_mask == 1, tasks_done, np.full(tasks_done.shape, -1))
        tasks_done[-1] = -1
        chosen_task = np.argmax(tasks_done)
        chosen_job = tasks[chosen_task].job_index
    return chosen_job


def random_task(tasks: List[Task], action_mask: np.array) -> int:
    """
    Returns a random task

    :param tasks: Not needed
    :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic

    :return: Index of the job selected according to the heuristic

    """

    chosen_job = None
    if np.sum(action_mask) == 1:
        chosen_job = np.argmax(action_mask)
    else:
        valid_values_0 = np.where(action_mask > 0)[0]

        if len(valid_values_0) > 2:
            chosen_job = np.random.choice(valid_values_0, size=1)[0]
        elif len(valid_values_0) == 0:
            print('this is not possible')
        else:
            chosen_job = np.random.choice(valid_values_0, size=1)[0]
    return chosen_job


def choose_random_machine(chosen_task, machine_mask) -> int:
    """
    Determines a random machine which is available according to the mask and chosen task. Useful for the FJSSP.

    :param chosen_task: ID of the task that is scheduled on the selected machine
    :param machine_mask: Machine mask from the environment that is to receive the machine action chosen by this function

    :return: Index of the chosen machine

    """
    machine_mask = np.array(np.where(machine_mask > 0))
    idx_valid_machine = np.where(machine_mask[0] == chosen_task)
    valid_machines = machine_mask[1][idx_valid_machine]
    chosen_machine = np.random.choice(valid_machines, size=1)[0]
    return chosen_machine


def choose_first_machine(chosen_task, machine_mask) -> int:
    """
    Determines the first (by index) machine which is available according to the mask and chosen task. Useful for the
    FJSSP

    :param chosen_task: ID of the task that is scheduled on the selected machine
    :param machine_mask: Machine mask from the environment that is to receive the machine action chosen by this function

    :return: Index of the chosen machine

    """
    machine_mask = np.array(np.where(machine_mask > 0))
    idx_valid_machine = np.where(machine_mask[0] == chosen_task)
    valid_machines = machine_mask[1][idx_valid_machine]
    return valid_machines[0]

class HeuristicSelectionAgent:
    """
    This class can be used to get the next task according to the heuristic passed as string abbreviation (e.g. EDD).
    If you want to edit a shortcut, or add one for your custom heuristic, adapt/extend the task_selection dict.

    :Example:

    .. code-block:: python

        def my_custom_heuristic():
            ...<function body>...

    or

    .. code-block:: python

        self.task_selections = {
            'rand': random_task,
            'XYZ': my_custom_heuristic
            }

    """

    def __init__(self) -> None:

        super().__init__()
        # Map heuristic ids to corresponding function
        self.task_selections = {
            'rand': random_task,
            'EDD': edd,
            'SPT': spt,
            'MTR': mtr,
            'LTR': ltr,
            'EDD_ASP': edd_asp,
            'SPT_ASP': spt_asp,
            'MPO_ASP': mpo_asp,
            'LPO_ASP': lpo_asp,
            'LPO_ROOT_ASP': lpo_root_asp,
            'MPO_ROOT_ASP': mpo_root_asp,
            'rand_asp': random_task_asp,
            'LETSA': letsa
        }

    def __call__(self, tasks: List, action_mask: np.array, task_selection: str, feasible_tasks = None, visited = None, max_deadline = None):
        """
        Selects the next heuristic function according to the heuristic passed as string abbreviation
        and the assignment in the task_selections dictionary

        :param tasks: List of task objects, so one instance
        :param action_mask: Action mask from the environment that is to receive the action selected by this heuristic
        :param task_selection: Heuristic string abbreviation (e.g. EDD)

        :return: Index of the job or task selected according to the heuristic

        """
        choose_task = self.task_selections[task_selection]

        chosen_task = None
        if task_selection == 'LETSA':
            chosen_task, completion_time = choose_task(tasks, action_mask, feasible_tasks, visited, max_deadline)
            return chosen_task, completion_time
        else:
            chosen_task = choose_task(tasks, action_mask)
            return chosen_task
