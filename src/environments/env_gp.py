from typing import List

from src.data_generator.task import Task
from src.models.machine import Machine
from src.environments.env_util import backward_planning_completion_time
from typing  import Any

class EnvGP():

    def __init__(self, config: dict, data: List[List[Task]], binary_features=None, from_test=False):
        """
        :param data: a list with the instances
        """
        self.seed = config.get('seed', None)
        self.np = config.get('no_parallel_processes', 1)
        self.evaluation_type = config.get('evaluation_type', 'best')
        #print("self.evaluation_type", self.evaluation_type)

        # import data containing all instances
        self.instances_no = len(data)
        self.current_instances: List[List[Task]] = data

        #current_instance
        self.current_instance_index = -1

        #test.py uses environment to go through current instance and schedule it
        #the following variables are set in order to make testing part run
        if from_test:
            self.get_next_instance() #only one instance is in instances list when environment is created from test
            self.tasks = self.operations

        #TODO this are add because of the test function
        self.done = False #the instance was scheduled
        self.tardiness = [0]
        self.action_history = []

    def get_next_instance(self):

        self.done = False
        self.current_instance_index += 1

        self.operations = self.current_instances[self.current_instance_index].copy()

        self.no_operations = len(self.operations)
        self.no_machines   = len(self.operations[0].machines)

        #print("self.no_operations", self.no_operations, "self.no_machines", self.no_machines)
        # for op in self.operations:
        #     print(self.operations)

        self.no_uncheduled_operations = self.no_operations
        self.no_used_machines = self.no_machines

        # structure to store the machine occupancy intervals
        self.machines = dict()
        for i in range(self.no_machines):
            self.machines[i] = Machine()
        self.machine_ready_time = [0] * self.no_machines
        self.machine_operation_no = [0] * self.no_machines #number of operations that can be scheduled on machine
        self.machine_queue_op_no = [0] * self.no_machines
        self.machine_queue_op_duration = [0] * self.no_machines

        # maps the task_id with the index in the tasks list
        self.operations_redy = []
        self.index_operation = dict()
        self.max_deadline = 0
        for op_idx, operation in enumerate(self.operations):
            #print(operation.task_index, operation.parent_index, operation.machines)

            operation.done = False #need because GP evaluates multiple times an instance
            operation.last_child_scheduled_finished = 0

            self.index_operation[operation.task_id] = op_idx
            self.max_deadline = operation.deadline if operation.deadline > self.max_deadline else self.max_deadline
            # operation/task status (-1 - unscheduled, 0 - scheduled,  1 - redy)
            #print(op_idx, "children", operation.children, len(operation.children), "parent", operation.parent_index)
            self.operations_redy.append(1 if len(operation.children) == 0 else -1)
            if len(operation.children) == 0:
                operation.release_time = 0
                for m_idx, elibigle_machine in enumerate(operation.machines):
                    if elibigle_machine:
                        self.machine_queue_op_no[m_idx] += 1
                        self.machine_queue_op_duration[m_idx] += operation.execution_times_setup[m_idx]

            #print(self.operations_redy)
            #print("operation.machines", operation.machines)
            for m_idx, elibigle_machine in enumerate(operation.machines):
                if elibigle_machine:
                    self.machine_operation_no[m_idx] += 1

        # print("new instance", self.current_instance_index, "op no", self.no_operations, "m no", self.no_machines, "deadline",  self.max_deadline)
        # print("redy op", [op_idx for op_idx, redy_op in enumerate(self.operations_redy) if redy_op == 1])
        # print("!!!get_next_instance-end", self.no_operations, self.no_machines,  self.max_deadline)
        # print(" self.operations_redy", self.operations_redy)
        # print("index_operation", self.index_operation)

    def get_heppler_informations(self):
        """
        For each machine:
        - counts number of operations that can be executed on machine and are not already scheduled
        - finds maximum execution time of operations that can be executed on machine and are not already scheduled

        :param self:
        :return: max_processing_times_per_machine, no_of_operations_executable_on_machine
        """
        max_processing_times_per_machine = [0] * self.no_machines
        no_of_operations_executable_on_machine = [0] * self.no_machines
        for _, operation in enumerate(self.operations):
            if not operation.done:
                for machine_idx in range(self.no_machines):
                    if operation.machines[machine_idx] == 1:
                        max_processing_times_per_machine[machine_idx] = max(max_processing_times_per_machine[machine_idx],
                                                                            operation.execution_times_setup[machine_idx])
                        no_of_operations_executable_on_machine[machine_idx] += 1
        return max_processing_times_per_machine, no_of_operations_executable_on_machine

    def get_next_action(self, priority_func: Any, individual_trees_no=1):
        if individual_trees_no == 1:
            return self.get_next_action_pair(priority_func)
        elif individual_trees_no == 2:
            return self.get_next_action_dispach_routing(priority_func)
        return None

    def get_next_action_dispach_routing(self, priority_func: Any):
        '''
        Selects the next pair to be scheduled
        :param priority_func: a GP priority rule or rules that selects a pair (op, machine)
        :return:
        '''
        ready_operations = []
        max_processing_times_per_machine, no_of_operations_executable_on_machine = self.get_heppler_informations()
        makespan = self.get_makespan()

        # print ("redy op", [op_idx for op_idx, redy_op in enumerate(self.operations_redy) if redy_op ==1])
        for op_idx, redy_op in enumerate(self.operations_redy):
            if redy_op == 1:
                operation = self.operations[op_idx]

                # operations features
                # 2. O_MeanPT- Mean processing time: Estimates operation duration.
                feat_op_mean_time = operation.average_execution_times_setup

                # 3. O_MinPT- Minimum processing time: Highlights the quickest possible execution time.
                feat_op_min_time = operation.min_execution_times_setup

                # 4. O_Flex - Ratio of machines that are eligible for Oij to total machine number
                no_eligible_machines = operation.machines.count(1)
                feat_op_flexibility_factor = float(no_eligible_machines) / self.no_machines
                # TODO vezi daca are sens self.no_used_machines sa fie la fel ca la GNN

                # 5.O_Path_OpNo
                feat_op_path_opNO = operation.no_remaining_operations

                # 6.O_Path_minLen
                feat_op_path_minLen = operation.remaining_work

                # 7.O_Ready
                feat_op_ready = makespan - operation.release_time

                score = 0
                l = []
                if self.evaluation_type == 'assemble':
                    for pf in priority_func:
                        _pf = pf[0]
                        a = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                feat_op_path_opNO, feat_op_path_minLen, feat_op_ready)
                        l.append(a)
                        score += a
                    score /= len(priority_func)
                elif self.evaluation_type == 'best' or self.evaluation_type == 'assemble-test':
                    _pf = priority_func[0]
                    score = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                feat_op_path_opNO, feat_op_path_minLen, feat_op_ready)
                # print(score,l)
                ready_operations.append((score, op_idx))
        ready_operations.sort(reverse=True)  # higher score first

        _, selected_operation_idx = ready_operations.pop(0)
        selected_operation = self.operations[selected_operation_idx]

        ready_machines = []
        for m_idx, eligible_machine in enumerate(selected_operation.machines):
            if eligible_machine == 1:
                processing_time = selected_operation.execution_times_setup[m_idx]

                # Features on operation-machine edges
                # 1. E_PT - Processing time p_{ik}  of operation i on machine k
                feat_edge_processing_time = processing_time

                # Machines features
                # 1. M_RT - Last operation completion time t_{last}: Determines machine availability.
                feat_machine_ready_time = self.machine_ready_time[m_idx]
                # 2. M_OP - Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
                feat_machine_operation_proportion = self.machine_operation_no[
                                                        m_idx] / self.no_uncheduled_operations if self.no_uncheduled_operations != 0 else 0
                # 3. M_UT - Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
                feat_machine_utilization_percentage = self.machines[m_idx].get_total_occupancy_duration() / \
                                                      makespan if makespan != 0 else 0
                # 4. M_QL
                feat_machine_queue_length = self.machine_queue_op_no[m_idx]
                # 5. M_QD
                feat_machine_queue_duration = self.machine_queue_op_duration[m_idx]
                # M_CT_A
                feat_machine_compleation_time_append = max(selected_operation.release_time, self.machine_ready_time[m_idx]) + processing_time
                # M_CT_B
                index, start_time, end_time = backward_planning_completion_time(selected_operation,
                                                                                               m_idx,
                                                                                               self.machines)

                feat_machine_compleation_time_backward = end_time

                # if feat_machine_compleation_time_append != feat_machine_compleation_time_backward:
                #     print('M_CT_A',feat_machine_compleation_time_append,'M_CT_b',feat_machine_compleation_time_backward,
                #           [[[task.started,task.finished] for task in self.machines[m_idx].intervals]], selected_operation.last_child_scheduled_finished, selected_operation.execution_times_setup[m_idx])

                # print("priority_func", priority_func)
                # print(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                #                       feat_machine_ready_time, feat_machine_operation_proportion,
                #                       feat_machine_utilization_percentage,
                #                       feat_edge_processing_time, feat_edge_PT_maxPT, feat_edge_PT_maxMachinePT)

                score = 0
                l = []
                if self.evaluation_type == 'assemble':
                    for pf in priority_func:
                        _pf = pf[1]
                        a = _pf(feat_edge_processing_time,
                                feat_machine_ready_time,
                                feat_machine_operation_proportion,
                                feat_machine_utilization_percentage,
                                feat_machine_queue_length,
                                feat_machine_queue_duration,
                                feat_machine_compleation_time_append,
                                feat_machine_compleation_time_backward)
                        l.append(a)
                        score += a
                    score /= len(priority_func)
                elif self.evaluation_type == 'best' or self.evaluation_type == 'assemble-test':
                    _pf = priority_func[1]
                    score = _pf(feat_edge_processing_time,
                                feat_machine_ready_time,
                                feat_machine_operation_proportion,
                                feat_machine_utilization_percentage,
                                feat_machine_queue_length,
                                feat_machine_queue_duration,
                                feat_machine_compleation_time_append,
                                feat_machine_compleation_time_backward
                                )
                # print(score,l)
                ready_machines.append((score, m_idx))

        # select pair
        ready_machines.sort(reverse=True)  # higher score first

        _, selected_machine_idx = ready_machines.pop(0)
        self.action_history.append((selected_operation_idx, selected_machine_idx))
        # print("op", selected_operation_idx, "m", selected_machine_idx)
        return (selected_operation_idx, selected_machine_idx)


    def get_next_action_pair(self, priority_func: Any, individual_trees_no=1):
        '''
        Selects the next pair to be scheduled
        :param priority_func: a GP priority rule or rules that selects a pair (op, machine)
        :return:
        '''
        ready_pairs = []
        max_processing_times_per_machine, no_of_operations_executable_on_machine = self.get_heppler_informations()
        makespan = self.get_makespan()

        #print ("redy op", [op_idx for op_idx, redy_op in enumerate(self.operations_redy) if redy_op ==1])
        for op_idx, redy_op in enumerate(self.operations_redy):
            if redy_op == 1:
                operation = self.operations[op_idx]

                # operations features
                # 2. O_MeanPT- Mean processing time: Estimates operation duration.
                feat_op_mean_time = operation.average_execution_times_setup

                # 3. O_MinPT- Minimum processing time: Highlights the quickest possible execution time.
                feat_op_min_time = operation.min_execution_times_setup

                # 4. O_Flex - Ratio of machines that are eligible for Oij to total machine number
                no_eligible_machines = operation.machines.count(1)
                feat_op_flexibility_factor = float(no_eligible_machines) / self.no_machines
                # TODO vezi daca are sens self.no_used_machines sa fie la fel ca la GNN

                #5.O_Path_OpNo
                feat_op_path_opNO = operation.no_remaining_operations

                #6.O_Path_minLen
                feat_op_path_minLen = operation.remaining_work

                #7.O_Ready
                feat_op_ready = makespan - operation.release_time
                for m_idx, eligible_machine in enumerate(operation.machines):
                    if eligible_machine == 1:
                        processing_time = operation.execution_times_setup[m_idx]

                        # Features on operation-machine edges
                        # 1. E_PT - Processing time p_{ik}  of operation i on machine k
                        feat_edge_processing_time = processing_time

                        # 2. E_PT_maxPT - Ratio of p_{ik} to the maximum processing time of p_{il}  l=1,M_i  (M_i= total number of machines on which op i can be executed)

                        feat_edge_PT_maxPT = processing_time / operation.max_execution_times_setup if operation.max_execution_times_setup > 0 else 0

                        # 3. E_PT_maxMPT - Ratio of p_{ik} to the maximum processing time of p_{lk}  l=1,N _k (N_k= total number of operations that can be executed on machine k)
                        feat_edge_PT_maxMachinePT = processing_time / max_processing_times_per_machine[m_idx] if \
                        max_processing_times_per_machine[m_idx] != 0 else 0

                        # Machines features
                        # 1. M_RT - Last operation completion time t_{last}: Determines machine availability.
                        feat_machine_ready_time = self.machine_ready_time[m_idx]
                        # 2. M_OP - Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
                        feat_machine_operation_proportion = self.machine_operation_no[
                                                                m_idx] / self.no_uncheduled_operations if self.no_uncheduled_operations != 0 else 0
                        # 3. M_UT - Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
                        feat_machine_utilization_percentage = self.machines[m_idx].get_total_occupancy_duration() / \
                                                              makespan if makespan != 0 else 0
                        #4. M_QL
                        feat_machine_queue_lenght = self.machine_queue_op_no[m_idx]
                        # 5. M_QD
                        feat_machine_queue_duration = self.machine_queue_op_duration[m_idx]

                        # print("priority_func", priority_func)
                        # print(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                        #                       feat_machine_ready_time, feat_machine_operation_proportion,
                        #                       feat_machine_utilization_percentage,
                        #                       feat_edge_processing_time, feat_edge_PT_maxPT, feat_edge_PT_maxMachinePT)

                        score = 0
                        l=[]
                        if self.evaluation_type == 'assemble':
                            for pf in priority_func:
                                _pf = pf[0]
                                a = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                                  feat_machine_ready_time, feat_machine_operation_proportion,
                                                  feat_machine_utilization_percentage,
                                                  feat_edge_processing_time, feat_edge_PT_maxPT,
                                                  feat_edge_PT_maxMachinePT,
                                                  feat_op_path_opNO, feat_op_path_minLen, feat_op_ready,
                                                  feat_machine_queue_lenght,feat_machine_queue_duration)
                                l.append(a)
                                score+=a
                            score /= len(priority_func)
                        elif self.evaluation_type == 'best' or self.evaluation_type == 'assemble-test':
                            _pf = priority_func[0]
                            score = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                          feat_machine_ready_time, feat_machine_operation_proportion,
                                          feat_machine_utilization_percentage,
                                          feat_edge_processing_time, feat_edge_PT_maxPT, feat_edge_PT_maxMachinePT,
                                          feat_op_path_opNO, feat_op_path_minLen, feat_op_ready,
                                          feat_machine_queue_lenght,feat_machine_queue_duration)
                        #print(score,l)
                        ready_pairs.append((score, op_idx, m_idx))
        # select pair
        ready_pairs.sort(reverse=True)  # higher score first

        _, selected_operation_idx, selected_machine_idx = ready_pairs.pop(0)
        self.action_history.append((selected_operation_idx, selected_machine_idx))
        #print("op", selected_operation_idx, "m", selected_machine_idx)
        return (selected_operation_idx, selected_machine_idx)

    def step(self, action, **args):
        '''
        Shedule operation on machine and updates the internal information
        :param action:
        :param args:
        :return:
        '''

        selected_operation_idx = action[0]
        selected_machine_idx = action[1]
        #print("op_idx", selected_operation_idx , "m_idx",action[1])


        # update operations structure
        selected_operation = self.operations[selected_operation_idx]

        #primul interval pe care încape
        index_machine_interval, start_time, completion_time = backward_planning_completion_time(selected_operation,
                                                                        selected_machine_idx,
                                                                        self.machines)
        #intervalul care se potriveste cel mai bine ?!

        #le calculeaza fct de mai sus - asta am modificat pt back insertion
        # start_time = max(selected_operation.last_child_scheduled_finished,
        #                  self.machine_ready_time[selected_machine_idx])  # FM max din cele 2
        # completion_time = start_time + selected_operation.execution_times_setup[selected_machine_idx]

        selected_operation.done = True
        selected_operation.started = start_time
        selected_operation.finished = completion_time
        selected_operation.selected_machine = selected_machine_idx
        selected_op_parent_idx = selected_operation.parent_index
        if selected_op_parent_idx is not None:
            selected_op_parent = self.operations[selected_op_parent_idx]

            #updare parent start time
            selected_op_parent.last_child_scheduled_finished = \
                max(completion_time, selected_op_parent.last_child_scheduled_finished)

            # add parent operation to available operations list
            parent_ready = True
            for op_idx in selected_op_parent.children:
                if not self.operations[op_idx].done:
                    parent_ready = False
            if parent_ready:
                self.operations_redy[self.index_operation[selected_op_parent.task_id]] = 1
                selected_op_parent.release_time = selected_op_parent.last_child_scheduled_finished #TODO: FM - 2 variabile care stocheaza acceasi informatie

        # print(' self.operations_redy',self.operations_redy)
        # print('Scheduled operation: ', selected_operation_idx, self.operations[selected_operation_idx].started,
        #       self.operations[selected_operation_idx].finished, self.operations[selected_operation_idx].selected_machine,
        #       self.operations[selected_operation_idx].done,self.operations[selected_operation_idx].parent_index)
        # if selected_op_parent_idx:
        #     print('  parent strat time', self.operations[selected_op_parent_idx].last_child_scheduled_finished)

        # update machines
        #self.machines[selected_machine_idx].add_last_interval(selected_operation) -inlocuit cu partea de insert before
        self.machines[selected_machine_idx].add_interval(index_machine_interval, selected_operation)


        # update internal structures
        self.machine_ready_time[selected_machine_idx] = completion_time
        self.operations_redy[selected_operation_idx] = 0
        self.no_uncheduled_operations -= 1

        for m_idx, eligible_machine in enumerate(selected_operation.machines):
            if eligible_machine:
                self.machine_operation_no[m_idx] -= 1  # eliminate opreation from the number of operations on machine
                self.machine_queue_op_no[m_idx] -= 1
                self.machine_queue_op_duration[m_idx] -= selected_operation.execution_times_setup[m_idx]
                if self.machine_operation_no[m_idx] == 0:
                    self.no_used_machines -= 1

        #print(" self.operations_redy", self.operations_redy)
        self.done = True if self.operations_redy.count(1) == 0 else False

        return 0, 0, self.done

    def evaluate_instance(self, priority_funcs: Any, get_next_action=1)->float:
            """
            :param priority_func: the function used to select next pair (operation, machine)
            :return:
            """
            #print("evaluate_instance", self.current_instance_index)
            #if self.current_instance_index != 0: #needed because for test.py the first instance need to be load in init function
            self.get_next_instance()
            #print("machine time:",self.machine_ready_time)
            #print("priority_funcs", priority_funcs)
            while not self.done:
                action = self.get_next_action(priority_funcs, get_next_action)
                self.step(action)
                #print("max(self.machine_ready_time)",max(self.machine_ready_time))

            return self.get_makespan() #makespan

    def get_makespan(self, *args):
        return max(self.machine_ready_time)  # makespan

