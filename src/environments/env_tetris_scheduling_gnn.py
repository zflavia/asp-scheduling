import numpy as np
from typing import List
import copy
import random
import heapq
from src.data_generator.task import Task
from src.environments.env_tetris_scheduling import Env
import torch_geometric.transforms as T
import numpy

from src.models.machine import Machine
from torch_geometric.data import HeteroData
import torch

class EnvGNN(Env):
    def __init__(self, config: dict, data: List[List[Task]], binary_features='1001011000'):

        super(EnvGNN, self).__init__(config, data, binary_features)

        self.generate_gnn()


    def generate_gnn(self):
        self.num_operations = sum(1 for task in self.tasks if not task.done)
        # print('num_operations', self.num_operations)

        self.heteroData["operation"].x = torch.zeros((self.num_operations, self.num_features_oper), dtype= torch.float)
        # print('self.heteroData["operation"].x', self.heteroData["operation"].x.shape)
        self.heteroData["machine"].x = torch.zeros((self.num_machines, self.num_features_mach), dtype= torch.float)

        machines_counter_dynamic = np.zeros(self.num_machines)
        for i in range(self.num_machines):
            machines_counter_dynamic[i] = 0

        aux_list_op = []
        aux_list_op_status = []
        aux_list_op_mean_processing_time = []
        aux_list_op_min_processing_time = []
        aux_list_op_proportion_machines = []
        aux_list_op_features = []

        aux_list_mach = []
        aux_list_mach_2 = []
        aux_list_features = []
        aux_list_number_of_operations_executable_on_machines = [0] * len(self.tasks[0].machines)
        aux_list_op_mach_processing_times = []
        aux_list_op_mach_processing_time_ratios_b = []
        aux_list_op_mach_processing_time_ratios_a = []
        max_processing_times_per_machine = [0] * self.num_machines

        # TODO: take the completion time from the dict of completion times of the machines dict as in the old code
        # TODO: calculate the actual ratio of the completion time of the last operation
        # DONE in the code below
        aux_list_last_operation_completion_time = []
        aux_list_utilization_percentage = []
        for machine in range(len(self.tasks[0].machines)):
            aux_list_last_operation_completion_time.append(self.ends_of_machine_occupancies[machine])
            # TODO: calculate the actual ratio of the completion time of the last operation
            # DONE in the code below
            total_occupancy_duration = self.machines[machine].get_total_occupancy_duration()
            if total_occupancy_duration == 0:
                aux_list_utilization_percentage.append(0)
            else:
                aux_list_utilization_percentage.append(total_occupancy_duration / self.ends_of_machine_occupancies[machine])

        aux_list_mach_features = []

        for task_i, task in enumerate(self.tasks):
            # TODO: only for those tasks that are not done yet
            # DONE in the code below
            if not task.done:
                # Operation-operation edges
                aux_list_op.append([task_i, task_i])
                # TODO: this can be improved by using the children list
                # DONE in the code below
                for task_j in task.children:
                    # TODO: this should be done the other way around, i.e. the child should be appended to the parent
                    # DONE in the code below
                    aux_list_op.append([task_j, task_i])

                # Operation-machine edges
                count_eligible_machines = task.machines.count(1)

                for machine in range(len(task.machines)):
                    if task.machines[machine] == 1:
                        aux_list_mach.append([task_i, machine])
                        aux_list_mach_2.append([machine, task_i])
                        # Features on operation-machine edges

                        # TODO: incorporate these values in a new field in Task class task.execution_times[machine] + task.setup_times[machine]
                        # DONE in the Task class
                        # 1. Processing time p_{ik}  of operation i on machine k
                        aux_list_op_mach_processing_times.append(task.execution_times_setup[machine])
                        # 2. Ratio of p_{ik} to the maximum processing time of p_{il}  l=1,M_i  (M_i= total number of machines on which op i can be executed)
                        # Compute the ratio of the processing time on the current machine to the maximum processing time
                        ratio = (task.execution_times_setup[machine]) / task.max_execution_times_setup
                        aux_list_op_mach_processing_time_ratios_b.append(ratio)
                        # 3. Ratio of p_{ik} to the maximum processing time of p_{lk}  l=1,N _k (N_k= total number of operations that can be executed on machine k)
                        max_processing_times_per_machine[machine] = max(max_processing_times_per_machine[machine], task.execution_times_setup[machine])

                        # b. Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
                        if not task.done:
                            aux_list_number_of_operations_executable_on_machines[machine] += 1

                # Operation features
                # a. for operations a binary indicator bo ∈ {0, 1} that indicates if the operation is ready.
                # TODO: 1 can be executable (children already scheduled, or no children), 0 otherwise
                # DONE in the code below
                if len(task.children) == 0:
                    aux_list_op_status.append(1)
                else:
                    is_executable = 1
                    for task_j in task.children:
                        if not self.tasks[task_j].done:
                            is_executable = 0
                            break
                    aux_list_op_status.append(is_executable)

                # aux_list_op_status.append(0)
                # if task.children == []:
                #     aux_list_op_status.append(1)
                # else:
                #     aux_list_op_status.append(0)
                # b. Mean processing time: Estimates operation duration.
                aux_list_op_mean_processing_time.append(task.average_execution_times_setup)
                # c. Minimum processing time: Highlights the quickest possible execution time.
                aux_list_op_min_processing_time.append(task.min_execution_times_setup)
                # d. Proportion of machines that are eligible for Oij
                # TODO: we need exactly the number of machines per problem instance
                aux_list_op_proportion_machines.append(count_eligible_machines / len(task.machines))

        # TODO: only for those tasks that are not done yet
        # DONE in the code below
        for task in self.tasks:
            if not task.done:
                for machine in range(len(task.machines)):
                    if task.machines[machine] == 1:
                        # 3. Ratio of p_{ik} to the maximum processing time of p_{lk}  l=1,N _k (N_k= total number of operations that can be executed on machine k)
                        ratio = (task.execution_times_setup[machine]) / max_processing_times_per_machine[machine]
                        aux_list_op_mach_processing_time_ratios_a.append(ratio)

        # Machines features
        # a. Last operation completion time t_{last}: Determines machine availability.
        # b. Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
        # c. Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
        # TODO: only count the operations that are not done yet, see TODO from above
        # DONE: in the code below
        aux_list_number_of_operations_executable_on_machines = [
             x / self.num_operations for x in aux_list_number_of_operations_executable_on_machines
        ]


        aux_list_mach_features = [aux_list_last_operation_completion_time, aux_list_number_of_operations_executable_on_machines, aux_list_utilization_percentage ]
        # print('before self.heteroData[machine].x', self.heteroData['machine'].x)
        self.heteroData['machine'].x = torch.Tensor(aux_list_mach_features).T
        # print('self.heteroData[machine].x', self.heteroData['machine'].x)


        # TODO: print this list of features to check if it shouldn't be transposed or not
        # DONE: it should have been transposed

        # TODO: print after every step to check if the graph is correct

        aux_list_op_features = [aux_list_op_status, aux_list_op_mean_processing_time, aux_list_op_min_processing_time, aux_list_op_proportion_machines]
        # print ('before self.heteroData[operation].x', self.heteroData['operation'].x)
        self.heteroData['operation'].x = torch.Tensor(aux_list_op_features).T
        # print ('self.heteroData[operation].x', self.heteroData['operation'].x)

        aux_list_features.append([aux_list_op_mach_processing_times, aux_list_op_mach_processing_time_ratios_a, aux_list_op_mach_processing_time_ratios_b])
        aux_list_features_flat = [item for sublist in aux_list_features for item in sublist]


        # print('before self.heteroData[operation, exec, machine].edge_index', self.heteroData['operation', 'exec', 'machine'].edge_index)
        self.heteroData['operation', 'prec', 'operation'].edge_index = torch.LongTensor(aux_list_op).T
        # print('self.heteroData[operation, prec, operation].edge_index', self.heteroData['operation', 'prec', 'operation'].edge_index)

        # print('before self.heteroData[operation, exec, machine].edge_index', self.heteroData['operation', 'exec', 'machine'].edge_index)
        self.heteroData['operation', 'exec', 'machine'].edge_index = torch.LongTensor(aux_list_mach).T
        # print('self.heteroData[operation, exec, machine].edge_index', self.heteroData['operation', 'exec', 'machine'].edge_index)
        # print('before self.heteroData[operation, exec, machine].edge_attr', self.heteroData['operation', 'exec', 'machine'].edge_attr)
        self.heteroData['operation', 'exec', 'machine'].edge_attr = torch.Tensor(aux_list_features_flat).T
        # print('self.heteroData[operation, exec, machine].edge_attr', self.heteroData['operation', 'exec', 'machine'].edge_attr)

        # print('before self.heteroData[machine, exec, operation].edge_index', self.heteroData['machine', 'exec', 'operation'].edge_index)
        self.heteroData['machine', 'exec', 'operation'].edge_index = torch.LongTensor(aux_list_mach_2).T
        # print('self.heteroData[machine, exec, operation].edge_index', self.heteroData['machine', 'exec', 'operation'].edge_index)
        # print('before self.heteroData[machine, exec, operation].edge_attr', self.heteroData['machine', 'exec', 'operation'].edge_attr)
        self.heteroData['machine', 'exec', 'operation'].edge_attr = torch.Tensor(aux_list_features_flat).T
        # print('self.heteroData[machine, exec, operation].edge_attr', self.heteroData['machine', 'exec', 'operation'].edge_attr)


    def reset (self):
        # update runs (episodes passed so far)
        self.runs += 1

        # reset episode counters and infos
        self.num_steps = 0
        self.makespan = 0
        self.ends_of_machine_occupancies = np.zeros(self.num_machines, dtype=int)
        #  kind of schedule dict with start_date and end_date
        self.machines = dict()
        self.machines_counter = dict()
        for i in range(self.num_machines):
            self.machines[i] = Machine()
            self.machines_counter[i] = 0

        self.action_history = []
        self.executed_job_history = []
        self.reward_history = []

        # clear episode rewards after all training data has passed once. Stores info across runs.
        if self.data_idx == 0:
            self.episodes_makespans, self.episodes_rewards, self.episodes_tardinesses = ([], [], [])
            self.iterations_over_data += 1

        # load new instance every run
        self.data_idx = self.runs % len(self.data)

        if self.sp_type == 'asp':
            self.num_jobs, self.num_tasks, self.max_runtime, self.max_deadline, self.max_sum_runtime_setup_pair = self.get_instance_info(self.data_idx)
            self.max_task_index: int = self.num_tasks - 1
            self.num_all_tasks: int = self.num_jobs * self.num_tasks
            self.tardiness: numpy.ndarray = np.zeros(self.num_all_tasks, dtype=int)
        self.tasks = copy.deepcopy(self.data[self.data_idx])
        self.task_job_mapping = {(task.job_index, task.task_index): i for i, task in enumerate(self.tasks)}

        # retrieve maximum deadline of the current instance
        max_deadline = max([task.deadline for task in self.tasks])
        self.max_deadline = max_deadline if max_deadline > 0 else 1

        self.generate_gnn()

        self.state = copy.deepcopy(self.heteroData)

        # self.calculate_next_state()
        self.calculate_mask()

        return self.state

    def compute_reward(self, use_letsa=False):
        """
        Calculates the reward that will later be returned to the agent. Uses the self.reward_strategy string to
        discriminate between different reward strategies. Default is 'dense_reward'.

        :return: Reward

        """
        if self.reward_strategy == 'dense_makespan_reward':
            # dense reward for makespan optimization according to https://arxiv.org/pdf/2010.12367.pdf
            reward = self.makespan - self.get_makespan(use_letsa)
            self.makespan = self.get_makespan(use_letsa)
        elif self.reward_strategy == 'sparse_makespan_reward':
            reward = self.sparse_makespan_reward(use_letsa)
        elif self.reward_strategy == 'mr2_reward':
            reward = self.mr2_reward()
        else:
            raise NotImplementedError(f'The reward strategy {self.reward_strategy} has not been implemented.')

        reward *= self.reward_scale

        return reward

    def step(self, action, **kwargs):

        print('before shape machine exec operation', self.state['machine', "exec", "operation"].edge_index.shape)
        print('before shape operation exec machine', self.state['operation', "exec", "machine"].edge_index.shape)
        print('before shape operation prec operation', self.state['operation', "prec", "operation"].edge_index.shape)
        # print('before shape machine exec operatio attr', self.state['operation', "prec", "operation"].edge_attr.shape)
        # print('before shape operation exec machine attr', self.state['operation', "exec", "machine"].edge_attr.shape)
        print('before shape operation', self.state['operation'].x.shape)
        print('before shape machine', self.state['machine'].x.shape)




        self.num_steps += 1

        action_mode = 'agent'  # set default, if the action mode is not defined assuming agent is taking it
        if 'action_mode' in kwargs.keys():
            action_mode = kwargs['action_mode']

        if action_mode == 'agent':
            action = self.state['machine', 'exec', 'operation'].edge_index[:,action]

            sel_op = action[1].item()
            sel_mach = action[0].item()
            print('sel_op', sel_op)
            print('sel_mach', sel_mach)

            mask = self.state['machine', 'exec', 'operation'].edge_index[1,:] != sel_op
            self.state['machine', 'exec', 'operation'].edge_index = self.state['machine', 'exec', 'operation'].edge_index[:,mask]
            # print('shape before', self.state['machine', 'exec', 'operation'].edge_attr.shape)
            self.state['machine', 'exec', 'operation'].edge_attr = self.state['machine', 'exec', 'operation'].edge_attr[mask]
            # print('shape after', self.state['machine', 'exec', 'operation'].edge_attr.shape)


            mask = self.state['operation', 'exec', 'machine'].edge_index[0,:] != sel_op
            self.state['operation', 'exec', 'machine'].edge_index = self.state['operation', 'exec', 'machine'].edge_index[:,mask]
            self.state['operation', 'exec', 'machine'].edge_attr = self.state['operation', 'exec', 'machine'].edge_attr[mask]

            print('before sel_op deletion from prec edges',  self.state['operation', 'prec', 'operation'].edge_index)
            self.state['operation', 'prec', 'operation'].edge_index = self.state['operation', 'prec', 'operation'].edge_index[:,self.state['operation', 'prec', 'operation'].edge_index[0,:] != sel_op]
            self.state['operation', 'prec', 'operation'].edge_index = self.state['operation', 'prec', 'operation'].edge_index[:,self.state['operation', 'prec', 'operation'].edge_index[1,:] != sel_op]
            print('after sel_op deletion from prec edges',  self.state['operation', 'prec', 'operation'].edge_index)

            self.state = T.RemoveIsolatedNodes()(self.state)


            # calculate new time
            start_time = self.ends_of_machine_occupancies[sel_mach]
            execution_setup_time  = self.tasks[sel_op].execution_times_setup[sel_mach]
            completion_time = start_time + execution_setup_time

            # schedule operation
            self.tasks[sel_op].done = True
            self.tasks[sel_op].started = start_time
            self.tasks[sel_op].finished = completion_time
            self.tasks[sel_op].selected_machine = sel_mach

            # update machine occupancy
            self.ends_of_machine_occupancies[sel_mach] = completion_time
            self.machines[sel_mach].add_last_interval(self.tasks[sel_op])


            # recalculate features for machine and operation nodes. order of features is
            # [aux_list_last_operation_completion_time, aux_list_number_of_operations_executable_on_machines, aux_list_utilization_percentage ]
            # [aux_list_op_status, aux_list_op_mean_processing_time, aux_list_op_min_processing_time, aux_list_op_proportion_machines]

            self.state["machine"].x[sel_mach, 0] = self.ends_of_machine_occupancies[sel_mach]
            self.state["machine"].x[sel_mach, 2] = self.machines[sel_mach].get_total_occupancy_duration() / self.ends_of_machine_occupancies[sel_mach]
            # aux_list_number_of_operations_executable_on_machines must be recalculated for all machines
            aux_list_number_of_operations_executable_on_machines = [0] * len(self.tasks[0].machines)
            aux_list_op_status = []

            # useful to know which row to delete from the operation node features
            sel_op_index_in_state = 0
            max_processing_times_per_machine = [0] * self.num_machines
            aux_list_op = []
            for task_i, task in enumerate(self.tasks):
                if not task.done:
                    # In case I decide to recompute the prec list onstead of deleting the selected op
                    # aux_list_op.append([task_i, task_i])
                    # for task_j in task.children:
                    #     if not self.tasks[task_j].done:
                    #         aux_list_op.append([task_j, task_i])

                    for machine in range(len(task.machines)):
                        if task.machines[machine] == 1:
                            aux_list_number_of_operations_executable_on_machines[machine] += 1
                            max_processing_times_per_machine[machine] = max(max_processing_times_per_machine[machine], task.execution_times_setup[machine])

                    if len(task.children) == 0:
                        aux_list_op_status.append(1)
                    else:
                        is_executable = 1
                        for task_j in task.children:
                            if not self.tasks[task_j].done:
                                is_executable = 0
                                break
                        aux_list_op_status.append(is_executable)
                # print('task.task_id', task.task_index, 'sel_op', sel_op)
                if not task.done:
                    if task_i < sel_op:
                        sel_op_index_in_state += 1

            # self.state['operation', 'prec', 'operation'].edge_index = torch.LongTensor(aux_list_op).T
            # print('aux_list_op', aux_list_op)
            self.state = T.RemoveIsolatedNodes()(self.state)


            self.num_operations -= 1 # decrease the number of operations by 1 because one operation has been scheduled
            # self.state["machine"].x[:, 1] = torch.tensor([x / self.num_operations for x in aux_list_number_of_operations_executable_on_machines], dtype=torch.float)
            for i in range(self.state["machine"].x.shape[0]):
                self.state["machine"].x[i, 1] = aux_list_number_of_operations_executable_on_machines[i] / self.num_operations


            # delete the sel_op row from the operation node features and then update the state with the op_status feature
            self.state["operation"].x = torch.cat((self.state["operation"].x[:sel_op_index_in_state], self.state["operation"].x[sel_op_index_in_state+1:]), dim=0)

            print('self.state["operation"].x.shape[0]', self.state["operation"].x.shape[0])
            for i in range(self.state["operation"].x.shape[0]):
                self.state["operation"].x[i, 0] = aux_list_op_status[i]

            # recalculate the operation-feature features
            # [aux_list_op_mach_processing_times, aux_list_op_mach_processing_time_ratios_a, aux_list_op_mach_processing_time_ratios_b]
            # the op-machine line was already deleted above
            aux_list_op_mach_processing_time_ratios_a = []
            for task in self.tasks:
                if not task.done:
                    for machine in range(len(task.machines)):
                        if task.machines[machine] == 1:
                            ratio = (task.execution_times_setup[machine]) / max_processing_times_per_machine[machine]
                            aux_list_op_mach_processing_time_ratios_a.append(ratio)


            print('before loop, .edge_attr.shape[0]', self.state['operation', "exec", "machine"].edge_attr.shape[0], 'len', len(aux_list_op_mach_processing_time_ratios_a))
            for i in range(self.state['operation', "exec", "machine"].edge_attr.shape[0]):
                # print('i', i, 'len aux_list_op_mach_processing_time_ratios_a', len(aux_list_op_mach_processing_time_ratios_a))
                self.state['operation', "exec", "machine"].edge_attr[i, 1] = aux_list_op_mach_processing_time_ratios_a[i]
            for i in range(self.state['machine', "exec", "operation"].edge_attr.shape[0]):
                self.state["machine", "exec", 'operation'].edge_attr[i, 1] = aux_list_op_mach_processing_time_ratios_a[i]

            done = self.check_done()
            self.calculate_mask()

            total_reward = self.compute_reward() / self.reward_normalization_factor
            infos = {'mask': self.state['machine', 'exec', 'operation'].mask}
            if done:
                return self.state, total_reward, done, infos

            if len(self.state['machine', 'exec', 'operation'].mask) - sum(self.state['machine', 'exec', 'operation'].mask)==1:
                self.state, reward, done, infos = self.step(self.sample(), action_mode='agent')
                total_reward += reward
            return self.state, total_reward, done, infos
        else:
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

            done = self.check_done()
            infos = {'mask': self.get_action_mask(is_asp=True)}
            if action_mode == 'heuristic' and self.sp_type == 'asp' and 'completion_time' in kwargs.keys():
                reward = self.compute_reward(use_letsa=True) / self.reward_normalization_factor
                infos = {'mask': self.get_action_mask(is_asp=True) }
            else:
                reward = self.compute_reward() / self.reward_normalization_factor
            return self.state, reward, done, infos


    def sample(self):
        return random.choice([i for i in range(len(self.state['machine', 'exec', 'operation'].mask)) if not self.state['machine', 'exec', 'operation'].mask[i]])


    def calculate_mask(self):
        k = 2

        # Select the first k (%) pairs (feasible op, mach) in increasing order of the completion time → to be used to set the action space

        pairs = []
        for task_i, task in enumerate(self.tasks):
            if not task.done:
                is_executable = False
                if  len(task.children) == 0:
                    is_executable = True
                else:
                    for task_j in task.children:
                        if not self.tasks[task_j].done:
                            is_executable = False
                            break
                if is_executable:
                    for machine_id in range(len(task.machines)):
                        if task.machines[machine_id] == 1:
                            pairs.append((task_i, machine_id, self.ends_of_machine_occupancies[machine_id] + task.execution_times_setup[machine_id]))


        res = [True] * self.state['machine', 'exec', 'operation'].edge_index.shape[1]

        sorted_pairs_with_indices = heapq.nsmallest(k, enumerate(pairs), key=lambda x: x[1][2])
        for index, _ in sorted_pairs_with_indices:
            res[index] = False

        self.state['machine', 'exec', 'operation'].mask = torch.BoolTensor(res)

    def normalize_state(self, state):
        state = copy.deepcopy(state)

        for i in range(state["operation"].x.shape[1]):
            state["operation"].x[:,i] = (2*(state["operation"].x[:,i] - state["operation"].x[:,i].min())/(state["operation"].x[:,i].max() - state["operation"].x[:,i].min() + 1e-7 )-1).float()

        for i in range(state["machine"].x.shape[1]):
            state["machine"].x[:,i] = (2*(state["machine"].x[:,i] - state["machine"].x[:,i].min())/(state["machine"].x[:,i].max() - state["machine"].x[:,i].min() + 1e-7 )-1).float()

        state[('operation', 'exec', 'machine')].edge_attr = (2*(state[('operation', 'exec', 'machine')].edge_attr -  state[('operation', 'exec', 'machine')].edge_attr.min())/(state[('operation', 'exec', 'machine')].edge_attr.max() - state[('operation', 'exec', 'machine')].edge_attr.min() + 1e-7 )-1).float()
        state[('machine', 'exec', 'operation')].edge_attr = (2*(state[('machine', 'exec', 'operation')].edge_attr -  state[('machine', 'exec', 'operation')].edge_attr.min())/(state[('machine', 'exec', 'operation')].edge_attr.max() - state['machine', 'exec', 'operation'].edge_attr.min() + 1e-7 )-1).float()
        return state




