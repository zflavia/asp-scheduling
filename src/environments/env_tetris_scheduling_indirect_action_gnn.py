import numpy as np
from typing import List

from src.data_generator.task import Task
from src.environments.env_tetris_scheduling import Env


from torch_geometric.data import HeteroData
import torch

class IndirectActionEnvGNN(Env):
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

        super(IndirectActionEnvGNN, self).__init__(config, data, binary_features)

        self.num_features_oper = 4
        self.num_features_mach = 3

        # TODO: num_operations must be always the number of operations that were not scheduled/done already!!!
        # DONE in the code below
        self.num_operations = sum(1 for task in self.tasks if not task.done)

        self.heteroData = HeteroData()

        self.generate_gnn()


    def generate_gnn(self):

        print('self.num_operations, self.num_features_oper', self.num_operations, self.num_features_oper)
        print('self.num_machines, self.num_features_mach', self.num_machines, self.num_features_mach)

        self.heteroData["operation"].x = torch.zeros((self.num_operations, self.num_features_oper), dtype= torch.float)
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

        # aux_list_last_operation_completion_time = [0] * len(self.tasks[0].machines)
        # aux_list_utilization_percentage = [0] * len(self.tasks[0].machines)

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
                    aux_list_op.append([task_i, task_j])

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

        print('before self.heteroData[machine].x', self.heteroData['machine'].x)
        self.heteroData['machine'].x = torch.Tensor(aux_list_mach_features).T
        print('self.heteroData[machine].x', self.heteroData['machine'].x)


        # TODO: print this list of features to check if it shouldn't be transposed or not
        # DONE: it should have been transposed

        # TODO: print after every step to check if the graph is correct

        aux_list_op_features = [aux_list_op_status, aux_list_op_mean_processing_time, aux_list_op_min_processing_time, aux_list_op_proportion_machines]
        print ('before self.heteroData[operation].x', self.heteroData['operation'].x)
        self.heteroData['operation'].x = torch.Tensor(aux_list_op_features).T
        print ('self.heteroData[operation].x', self.heteroData['operation'].x)

        aux_list_features.append([aux_list_op_mach_processing_times, aux_list_op_mach_processing_time_ratios_a, aux_list_op_mach_processing_time_ratios_b])

        # print('before self.heteroData[operation, exec, machine].edge_index', self.heteroData['operation', 'exec', 'machine'].edge_index)
        self.heteroData['operation', 'prec', 'operation'].edge_index = torch.LongTensor(aux_list_op).T
        print('self.heteroData[operation, prec, operation].edge_index', self.heteroData['operation', 'prec', 'operation'].edge_index)

        # print('before self.heteroData[operation, exec, machine].edge_index', self.heteroData['operation', 'exec', 'machine'].edge_index)
        self.heteroData['operation', 'exec', 'machine'].edge_index = torch.LongTensor(aux_list_mach).T
        print('self.heteroData[operation, exec, machine].edge_index', self.heteroData['operation', 'exec', 'machine'].edge_index)
        # print('before self.heteroData[operation, exec, machine].edge_attr', self.heteroData['operation', 'exec', 'machine'].edge_attr)
        self.heteroData['operation', 'exec', 'machine'].edge_attr = torch.Tensor(aux_list_features)
        print('self.heteroData[operation, exec, machine].edge_attr', self.heteroData['operation', 'exec', 'machine'].edge_attr)

        # print('before self.heteroData[machine, exec, operation].edge_index', self.heteroData['machine', 'exec', 'operation'].edge_index)
        self.heteroData['machine', 'exec', 'operation'].edge_index = torch.LongTensor(aux_list_mach_2).T
        print('self.heteroData[machine, exec, operation].edge_index', self.heteroData['machine', 'exec', 'operation'].edge_index)
        # print('before self.heteroData[machine, exec, operation].edge_attr', self.heteroData['machine', 'exec', 'operation'].edge_attr)
        self.heteroData['machine', 'exec', 'operation'].edge_attr = torch.Tensor(aux_list_features)
        print('self.heteroData[machine, exec, operation].edge_attr', self.heteroData['machine', 'exec', 'operation'].edge_attr)




