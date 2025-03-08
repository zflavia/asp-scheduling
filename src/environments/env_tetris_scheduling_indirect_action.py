import numpy as np
import copy
from typing import List
from gym import spaces
from numpy import ndarray

from src.data_generator.task import Task
from src.environments.env_tetris_scheduling import Env
from src.models.machine import Machine
from src.models.setqueue import SetQueue

class IndirectActionEnv(Env):
    """
    Scheduling environment for scheduling optimization according to
    https://www.sciencedirect.com/science/article/pii/S0952197622001130.

    Main differences to the vanilla environment:

    - ACTION: Indirect action mapping
    - REWARD: m-r2 reward (which means we have to train on the same data again and again)
    - OBSERVATION: observation different ("normalization" looks like division by max to [0, 1] in paper code). Not every
      part makes sense, due to the different interaction logic
    - INTERACTION LOGIC WARNING:
    - original paper: time steps are run through, the agent can take as many actions as it wants per time-step,
      but may not schedule into the past.
    - our adaptation: we still play tetris, meaning that we schedule whole blocks of work at a time

    :param config: Dictionary with parameters to specify environment attributes
    :param data: Scheduling problem to be solved, so a list of instances

    """
    def __init__(self, config: dict, data: List[List[Task]], binary_features='1001011000'):

        super(IndirectActionEnv, self).__init__(config, data, binary_features)

        self.action_space: spaces.Discrete = spaces.Discrete(self.num_tasks)

        # overwrite observation space
        observation_shape = np.array(self.state_obs).shape
        self.observation_space = spaces.Box(low=-1, high=1, shape=observation_shape)
        self.should_use_machine_task_pair = False
        self.should_determine_task_index = True


    #         for task in self.tasks:
    #             min_runtime = np.inf
    #             for machine_id in range(len(task.machines)):
    #                 if task.machines[machine_id] == 1:
    #                     min_runtime = min(min_runtime, task.execution_times[machine_id])
    #             task.runtime = min_runtime


    def step(self, action: int, **kwargs):
        """
        Step Function

        :param action: Action to be performed on the current state of the environment
        :param kwargs: should include "action_mode", because the interaction pattern between heuristics and
            the agent are different and need to be processed differently

        :return: Observation, reward, done, infos

        """
        # check if action_mode was set
        action_mode = 'agent'  # set default, if the action mode is not defined assuming agent is taking it
        if 'action_mode' in kwargs.keys():
            action_mode = kwargs['action_mode']

        selected_task_id, selected_machine = None, None
        if action_mode == 'agent':
            # get selected action via indirect action mapping
            next_tasks = self.get_next_tasks()
            if self.should_use_machine_task_pair == False and self.should_determine_task_index == False:
                #  search the closest avr_runtime from next tasks and then scale it.
                # this should return the index and use it later when executing the action
                #DZ: min-max scaling
                next_runtimes = copy.deepcopy([task.runtime if task is not None else np.inf for task in next_tasks])
                #next_runtimes = np.array(next_runtimes) / self.max_runtime
                next_runtimes = np.array(next_runtimes)
                if max(next_runtimes)-min(next_runtimes) != 0:
                    next_runtimes = (next_runtimes-min(next_runtimes))/(max(next_runtimes)-min(next_runtimes))
                else:
                    next_runtimes /= max(next_runtimes)
                #print("Step next_runtimes=",next_runtimes)
                #  this must be modified: for loop tasks and choose the min diff between the runtime si action
                min_diff = np.inf
                min_index = 0
                for i in range(len(next_tasks)):
                    if next_runtimes[i] != np.inf:
                        diff = abs(next_runtimes[i] - (action/9))
                        if diff < min_diff:
                            min_diff = diff
                            min_index = next_tasks[i].task_index
                action = min_index
            elif self.should_use_machine_task_pair == True and self.should_determine_task_index == False:
                min_diff = np.inf

                for task in next_tasks:
                    for machine_index in range(len(task.machines)):
                        if task.machines[machine_index] == 1:
                            term_one = (task.execution_times[machine_index] + task.setup_times[machine_index]) / self.max_sum_runtime_setup_pair
                            term_two = (action/9)
                            diff = abs(term_one - term_two)
                            if diff < min_diff:
                                min_diff = diff
                                selected_task_id = task.task_index
                                selected_machine = machine_index
            elif self.should_use_machine_task_pair == False and self.should_determine_task_index == True:
                min_diff = np.inf
                for task in next_tasks:
                    diff = abs(task.task_index - action)
                    if diff < min_diff:
                        min_diff = diff
                        selected_task_id = task.task_index
                    elif diff == min_diff:
                        pass
        else:
            # action remains the same
            pass

        # transform and store action
        selected_task_vector = None
        selected_job_vector = None
        if self.sp_type == 'asp':
            selected_task_vector = self.to_one_hot(action, self.num_tasks)
            self.action_history.append(action)
        else:
            selected_job_vector = self.to_one_hot(action, self.num_jobs)
            self.action_history.append(action)


        #  check if sp_type = asp to assign given task_idx from args to selected_task_id
        if action_mode == 'heuristic' and self.sp_type == 'asp':
            #  check if task_idx was set in args from heuristics
            selected_task_id = -1
            if 'task_idx' in kwargs.keys():
                selected_task_id = kwargs['task_idx']
            selected_task = self.get_selected_task_by_idx(selected_task_id)
            if not 'completion_time' in kwargs.keys():
                selected_machine = self.choose_machine(selected_task)
                #  job = 0 since we only have one job
                self.execute_action(0, selected_task, selected_machine)
            else:
                #  LETSA specific
                original_completion_time = kwargs['completion_time']
                machine_id, start_time, end_time, index = self.choose_machine_using_completion_time(selected_task, original_completion_time)
                #  job = 0 since we only have one job
                self.execute_action_with_given_interval(0, selected_task, machine_id, start_time, end_time, index)
        #  since  we select the task instead of job, then we need to get the machine directly as in ASP
        elif  action_mode == 'agent' and self.sp_type == 'asp' and self.should_use_machine_task_pair == True and self.should_determine_task_index == False:
            self.execute_action(0, self.tasks[selected_task_id], selected_machine)
        #  check if the task is a valid one (not planned and his children all planned)
        elif action_mode == 'agent' and self.sp_type == 'asp' and self.should_use_machine_task_pair == False and self.should_determine_task_index == True:
            selected_task = self.get_selected_task_by_idx(selected_task_id)
            selected_machine = self.choose_machine(selected_task)
            #  job = 0 since we only have one job
            self.execute_action(0, selected_task, selected_machine)
        # check if the action is valid/executable
        elif self.check_valid_job_action(selected_job_vector, self.last_mask):
            # if the action is valid/executable/schedulable
            selected_task_id, selected_task = self.get_selected_task(action)
            selected_machine = self.choose_machine(selected_task)
            self.execute_action(action, selected_task, selected_machine)
        else:
            # if the action is not valid/executable/scheduable
            pass

        # update variables and track reward
        #  action_mask either job_mask or task_mask
        if self.sp_type == 'asp':
            action_mask = self.get_action_mask(is_asp=True)
        else:
            action_mask = self.get_action_mask()
        infos = {'mask': action_mask}
        observation = self.state_obs
        if action_mode == 'heuristic' and self.sp_type == 'asp' and 'completion_time' in kwargs.keys():
            reward = self.compute_reward(use_letsa=True) / self.reward_normalization_factor
        else:
            reward = self.compute_reward() / self.reward_normalization_factor
        self.reward_history.append(reward)

        done = self.check_done()
        if done:
            episode_reward_sum = np.sum(self.reward_history)
            if action_mode == 'heuristic' and self.sp_type == 'asp' and 'completion_time' in kwargs.keys():
                makespan = self.get_makespan(use_letsa=True)
            else:
                makespan = self.get_makespan()
            tardiness = self.calculate_tardiness()

            self.episodes_makespans.append(makespan)
            self.episodes_rewards.append(np.mean(self.reward_history))

            self.logging_rewards.append(episode_reward_sum)
            self.logging_makespans.append(makespan)
            self.logging_tardinesses.append(tardiness)

            if self.runs % self.log_interval == 0:
                self.log_intermediate_step()
        self.num_steps += 1
        return observation, reward, done, infos

    def reset(self) -> ndarray:
        """
        - Resets the episode information trackers
        - Updates the number of runs
        - Loads new instance

        :return: First observation by calling the class function self.state_obs

        """
        # update runs (episodes passed so far)
        self.runs += 1

        # reset episode counters and infos
        self.num_steps = 0
        self.tardiness = np.zeros(self.num_all_tasks, dtype=int)
        self.makespan = 0
        self.ends_of_machine_occupancies = np.zeros(self.num_machines, dtype=int)
        #  kind of schedule dict with start_date and end_date
        self.machines = dict()
        self.machines_counter = dict()
        for i in range(self.num_machines):
            self.machines[i] = Machine()
            self.machines_counter[i] = 0

        self.tool_occupancies = [[] for _ in range(self.num_tools)]
        self.job_task_state = np.zeros(self.num_jobs, dtype=int)
        self.action_history = []
        self.executed_job_history = []
        self.reward_history = []

        # clear episode rewards after all training data has passed once. Stores info across runs.
        if self.data_idx == 0:
            self.episodes_makespans, self.episodes_rewards, self.episodes_tardinesses = ([], [], [])
            self.iterations_over_data += 1

        # load new instance every run
        self.data_idx = self.runs % len(self.data)
        # recompute self.num_tasks, self.max_runtime, self.max_deadline for ASP case
        if self.sp_type == 'asp':
            self.num_jobs, self.num_tasks, self.max_runtime, self.max_deadline, self.max_setup_time, self.max_sum_runtime_setup_pair = self.get_instance_info(self.data_idx)
            self.max_task_index: int = self.num_tasks - 1
            self.num_all_tasks: int = self.num_jobs * self.num_tasks
            self.tardiness: numpy.ndarray = np.zeros(self.num_all_tasks, dtype=int)
        self.tasks = copy.deepcopy(self.data[self.data_idx])
        if self.shuffle:
            np.random.shuffle(self.tasks)
        self.task_job_mapping = {(task.job_index, task.task_index): i for i, task in enumerate(self.tasks)}

        # retrieve maximum deadline of the current instance
        max_deadline = max([task.deadline for task in self.tasks])
        self.max_deadline = max_deadline if max_deadline > 0 else 1
        self.critical_path = ([], 0)

        return self.state_obs

    def is_leaf(self, task_index):
        return len(self.tasks[task_index].children) == 0 and self.tasks[task_index].parent_index != None

    def compute_paths(self, task_index, path, duration):
        path.append(task_index)

        # Compute the length (cumulative processing # time) of each path determined in step 4.1.
        if not self.tasks[task_index].done:
            duration += self.tasks[task_index].max_execution_times_setup
        else:
            duration += (self.tasks[task_index].finished - self.tasks[task_index].started)

        # 4.3 a: Determine the critical (the largest cumulative processing time) path
        if self.is_leaf(task_index) and duration > self.critical_path[1]:
            self.critical_path = (copy.deepcopy(path), duration)
            return
        for index_subtask in self.tasks[task_index].children:
            self.compute_paths(index_subtask, path, duration)
            path.pop()

    @property
    def state_obs(self) -> ndarray:
        """
        Transforms state (task state and factory state) to gym obs
        Scales the values between 0-1 and transforms to onehot encoding
        Confer https://www.sciencedirect.com/science/article/pii/S0952197622001130 section 4.2.1

        :return: Observation

        """

        next_tasks = self.get_next_tasks()

        features = {}
        #  sum of all task processing times still to be processed on each machine
        features['remaining_processing_times_on_machines'] = np.zeros(self.num_machines)
        features['task_status'] =  np.zeros(len(self.tasks))
        # estimated processing time per operation (EPT),
        features['operation_time_per_tasks'] = np.zeros(len(self.tasks))
        #  estimated completion time per operation
        features['completion_time_per_task'] =  np.zeros(len(self.tasks))
        #  estimated remaining processing time per operation
        features['estimated_remaining_processing_time_per_task'] = np.zeros(len(self.tasks))
        #  estimated) processing time for the next operation = estimated processing time for the next operation (successor of the current operation
        features['estimated_remaining_processing_time_per_successor_task'] = np.zeros(len(self.tasks))
        #         assignations =  np.zeros(len(self.tasks))
        #  remaining operations count = the number of operations on the branch from the current node(operation) to the roo
        features['remaining_tasks_count'] = np.zeros(len(self.tasks))
        #  assignations
        features['mat_machine_op'] = np.zeros((len(self.tasks), self.num_machines))

        #  machine bottleneck feature = ”number of unscheduled operations per machine” or ”total duration (sum of processing times) of unscheduled operations per machine
        #  mapping of many machines are used dynamically
        features['machines_counter_dynamic'] = np.zeros(self.num_machines)
        for i in range(self.num_machines):
            features['machines_counter_dynamic'][i] = 0
        for task in self.tasks:
            if not task.done:
                for index in range(len(task.machines)):
                    if task.machines[index] == 1:
                        features['machines_counter_dynamic'][index] += 1

        #  variables for computing the critical_path
        feasible_tasks = SetQueue()
        features['is_task_in_critical_path'] = np.zeros(len(self.tasks))
        self.critical_path = ([], 0)
        for task in self.tasks:
            if not task.parent_index:
                feasible_tasks.put(task.task_index)
        start_task_index = feasible_tasks.get()
        # 4.1 For each operation in the feasible list formulate all possible network paths.
        self.compute_paths(start_task_index, [], 0)
        for index in self.critical_path[0]:
            features['is_task_in_critical_path'][index] = 1

        for task in self.tasks:
            for index in range(len(task.machines)):
                features['mat_machine_op'][task.task_index][index]  = task.machines[index]
            if task.done:
                features['task_status'][task.task_index] = 0
                features['operation_time_per_tasks'][task.task_index] = task.finished - task.started
                features['completion_time_per_task'][task.task_index] = task.finished
            if not task.done:
                # complete with the specific execution time per machine not with task.runtime
                for index in range(len(task.machines)):
                    if task.machines[index] == 1:
                        features['remaining_processing_times_on_machines'][index] += task.execution_times[index]

                weight_up = 0
                weight_down = 0
                for index in range(len(task.machines)):
                    if task.machines[index] == 1:
                        #                         weight_up += (self.machines_counter[index] * task.execution_times[index])
                        #                         weight_down += self.machines_counter[index]
                        weight_up += (features['machines_counter_dynamic'][index] * task.execution_times[index])
                        weight_down += features['machines_counter_dynamic'][index]
                weighted_average_runtime = weight_up / weight_down

                # first assume that the task is not yet part of the next task that can be scheduled right away
                features['task_status'][task.task_index] = 1
                for next_task in next_tasks:
                    if next_task.task_index == task.task_index:
                        features['task_status'][task.task_index] = 0.5
                features['operation_time_per_tasks'][task.task_index] = weighted_average_runtime

                max_completion_time_per_child = 0
                for child_index in task.children:
                    for tasks_j in self.tasks:
                        if child_index == tasks_j.task_index:
                            max_completion_time_per_child = max(max_completion_time_per_child, features['completion_time_per_task'][tasks_j.task_index])
                features['completion_time_per_task'][task.task_index] = max_completion_time_per_child + features['operation_time_per_tasks'][task.task_index]

        # estimated_remaining_processing_time_per_task
        for task in self.tasks:
            features['estimated_remaining_processing_time_per_task'][task.task_index] = features['operation_time_per_tasks'][task.task_index]
            task_successor_index = task.parent_index

            # NPT: only for next successor:
            if task_successor_index is not None:
                features['estimated_remaining_processing_time_per_successor_task'][task.task_index] = features['operation_time_per_tasks'][task_successor_index]
            while task_successor_index is not None:
                features['estimated_remaining_processing_time_per_task'][task.task_index] += features['operation_time_per_tasks'][task_successor_index]
                if not task.done:
                    features['remaining_tasks_count'][task.task_index] += 1
                task_successor_index = self.tasks[task_successor_index].parent_index

        # normalization and flattening if needed
        # minmax scalar (x - min) / (max - min)
        if max(features['operation_time_per_tasks']) - min(features['operation_time_per_tasks']) != 0:
            features['operation_time_per_tasks'] = (features['operation_time_per_tasks'] - min(features['operation_time_per_tasks'])) / (max(features['operation_time_per_tasks']) - min(features['operation_time_per_tasks']))
        else:
            features['operation_time_per_tasks'] /= max(features['operation_time_per_tasks'])

        if max(features['remaining_processing_times_on_machines']) - min(features['remaining_processing_times_on_machines']) != 0:
            features['remaining_processing_times_on_machines'] = (features['remaining_processing_times_on_machines'] - min(features['remaining_processing_times_on_machines'])) / (max(max(features['remaining_processing_times_on_machines']), 1) - min(features['remaining_processing_times_on_machines']))
        else:
            features['remaining_processing_times_on_machines'] /= max(max(features['remaining_processing_times_on_machines']), 1)

        if max(features['completion_time_per_task']) - min(features['completion_time_per_task']) != 0:
            features['completion_time_per_task'] = (features['completion_time_per_task'] - min(features['completion_time_per_task'])) / (max(features['completion_time_per_task']) - min(features['completion_time_per_task']) )
        else:
            features['completion_time_per_task'] /= max(features['completion_time_per_task'])

        if max(features['estimated_remaining_processing_time_per_task']) - min(features['estimated_remaining_processing_time_per_task']) != 0:
            features['estimated_remaining_processing_time_per_task'] = (features['estimated_remaining_processing_time_per_task'] - min(features['estimated_remaining_processing_time_per_task'])) / (max(features['estimated_remaining_processing_time_per_task']) - min(features['estimated_remaining_processing_time_per_task']))
        else:
            features['estimated_remaining_processing_time_per_task'] /= max(features['estimated_remaining_processing_time_per_task'])
        features['mat_machine_op'] = features['mat_machine_op'].flatten()

        observation = []
        for i in range(len(self.binary_features)):
            if self.binary_features[i] == '1' and features[self.feature_index_mapping[i]] is not None:
                observation.append(features[self.feature_index_mapping[i]])
        observation = np.concatenate(observation)

        # print('observation:', observation)
        self._state_obs = observation
        return self._state_obs

    def check_valid_task_action(self, task_index):
        done = True
        for sub_task_index in self.tasks[task_index].children:
            if not self.tasks[sub_task_index].done:
                done = False
                break
        if done == True and not self.tasks[task_index].done:
            return True
        return False


    def get_action_mask(self, is_asp=False) -> np.array:
        """
        Get Action mask
        In this environment, we always treat all actions as valid, because the interaction logic accepts it. Note that
        we only allow non-masked algorithms.
        The heuristics, however, still need the job mask OR task_mask.
        0 -> available
        1 -> not available

        :return: Action mask

        """
        if is_asp == False:
            job_mask = np.where(self.job_task_state < self.num_tasks,
                                np.ones(self.num_jobs, dtype=int), np.zeros(self.num_jobs, dtype=int))
            self.last_mask = job_mask
            return job_mask
        else:
            task_mask = [0] * self.num_tasks
            for task in self.tasks:
                done = True
                for sub_task_index in task.children:
                    if not self.tasks[sub_task_index].done:
                        done = False
                if done == True and not task.done:
                    task_mask[task.task_index] = 1

            return task_mask


    def get_next_tasks(self):
        """returns the next tasks that can be scheduled. in case of asp, it returns the tasks whose children are all done and can be scheduled"""
        next_tasks = []
        if self.sp_type == 'asp':
            for task in self.tasks:
                done = True
                for sub_task_index in task.children:
                    if not self.tasks[sub_task_index].done:
                        done = False
                if done == True and not task.done:
                    next_tasks.append(self.tasks[task.task_index])
        else:
            for job in range(self.num_jobs):
                if self.job_task_state[job] >= self.num_tasks:  # means that job is done
                    next_tasks.append(None)
                else:
                    task_position = self.task_job_mapping[(job, self.job_task_state[job])]
                    next_tasks.append(self.tasks[task_position])
        return next_tasks
