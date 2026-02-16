from typing import List

from src.agents.gp.gp_common import GPBase
from src.data_generator.task import Task
from src.models.machine import Machine
from src.environments.env_util import backward_planning_completion_time
from typing  import Any

class EnvGP():

    def __init__(self, config: dict, data: List[List[Task]], binary_features=None, from_test: bool=False):
        """

        :param config: configuration dictionary
        :param data: a list with the test instances
        :param binary_features: not used
        :param from_test: boolean used to differentiate if it called from train aor test process
        """
        self.seed = config.get('seed', None)
        self.np = config.get('no_parallel_processes', 1)
        self.evaluation_type = GPBase.RuleEvaluationType(config.get('evaluation_type', 'best'))

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

        self.done = False #the instance was scheduled
        self.tardiness = [0]
        self.action_history = []

    def get_next_instance(self):

        self.done = False
        self.current_instance_index += 1

        self.operations = self.current_instances[self.current_instance_index].copy()

        self.no_operations = len(self.operations)
        self.no_machines   = len(self.operations[0].machines)

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

            operation.done = False #need because GP evaluates multiple times an instance
            operation.last_child_scheduled_finished = 0

            self.index_operation[operation.task_id] = op_idx
            self.max_deadline = operation.deadline if operation.deadline > self.max_deadline else self.max_deadline

            self.operations_redy.append(1 if len(operation.children) == 0 else -1)
            if len(operation.children) == 0:
                operation.release_time = 0
                for m_idx, elibigle_machine in enumerate(operation.machines):
                    if elibigle_machine:
                        self.machine_queue_op_no[m_idx] += 1
                        self.machine_queue_op_duration[m_idx] += operation.execution_times_setup[m_idx]

            for m_idx, elibigle_machine in enumerate(operation.machines):
                if elibigle_machine:
                    self.machine_operation_no[m_idx] += 1


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
            return self.get_next_action_1tree(priority_func)
        elif individual_trees_no == 2:
            return self.get_next_action_2trees(priority_func)
            #return self.get_next_action_2trees_normalized_features(priority_func)
        return None

    def get_next_action_2trees(self, priority_func: Any):
        '''
        Selects the next pair (operation, machine) to be scheduled
        :param priority_func: a GP priority rule or rules that selects a pair (op, machine)
        :return:
        '''
        ready_operations = []
        max_processing_times_per_machine, no_of_operations_executable_on_machine = self.get_heppler_informations()
        makespan = self.get_makespan()

        normalisation_dict={}

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

                # 5.O_Path_OpNo
                feat_op_path_opNO = operation.no_remaining_operations

                # 6.O_Path_minLen
                feat_op_path_minLen = operation.remaining_work

                # 7.O_Ready
                feat_op_ready = makespan - operation.release_time

                score = 0
                l = []
                if self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE:
                    for pf in priority_func:
                        _pf = pf[0]
                        a = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                feat_op_path_opNO, feat_op_path_minLen, feat_op_ready)
                        l.append(a)
                        score += a
                    score /= len(priority_func)
                elif (self.evaluation_type is GPBase.RuleEvaluationType.BEST or
                      self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE_INSTANCES):
                    _pf = priority_func[0]
                    score = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                feat_op_path_opNO, feat_op_path_minLen, feat_op_ready)

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

                score = 0
                l = []
                if self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE:
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
                elif (self.evaluation_type is GPBase.RuleEvaluationType.BEST or
                      self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE_INSTANCES):
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

                ready_machines.append((score, m_idx))

        # select pair
        ready_machines.sort(reverse=True)  # higher score first

        _, selected_machine_idx = ready_machines.pop(0)
        self.action_history.append((selected_operation_idx, selected_machine_idx))

        return (selected_operation_idx, selected_machine_idx)


    def max_min_scale_of_feature_values(self, features: list[str],
                                        data: dict[int, dict[str, float]]):
        mins = {f: float("inf") for f in features}
        maxs = {f: float("-inf") for f in features}

        for row in data.values():
            for f in features:
                x = row[f]
                if x < mins[f]: mins[f] = x
                if x > maxs[f]: maxs[f] = x

        for row in data.values():
            for f in features:
                lo, hi = mins[f], maxs[f]
                row[f] = (row[f] - lo) / (hi - lo) if hi > lo else 0.0

    def get_next_action_2trees_normalized_features(self, priority_func: Any):
        '''
        Selects the next pair (operation, machine) to be scheduled
        :param priority_func: a GP priority rule or rules that selects a pair (op, machine)
        :return:
        '''
        ready_operations = []
        #max_processing_times_per_machine, no_of_operations_executable_on_machine = self.get_heppler_informations()
        makespan = self.get_makespan()

        normalisation_dict = {}
        #compute and store features for all available operations
        for op_idx, redy_op in enumerate(self.operations_redy):
            if redy_op == 1:
                operation = self.operations[op_idx]
                operation_features_dict ={}

                # operations features
                # 2. O_MeanPT- Mean processing time: Estimates operation duration.
                feat_op_mean_time = operation.average_execution_times_setup
                operation_features_dict['O_MeanPT'] = feat_op_mean_time

                # 3. O_MinPT- Minimum processing time: Highlights the quickest possible execution time.
                feat_op_min_time = operation.min_execution_times_setup
                operation_features_dict['O_MinPT'] = feat_op_min_time

                # 4. O_Flex - Ratio of machines that are eligible for Oij to total machine number
                no_eligible_machines = operation.machines.count(1)
                feat_op_flexibility_factor = float(no_eligible_machines) / self.no_machines
                operation_features_dict['O_Flex'] = feat_op_flexibility_factor

                # 5.O_Path_OpNo
                feat_op_path_opNO = operation.no_remaining_operations
                operation_features_dict['O_Path_OpNo'] = feat_op_path_opNO

                # 6.O_Path_minLen
                feat_op_path_minLen = operation.remaining_work
                operation_features_dict['O_Path_minLen'] = feat_op_path_minLen

                # 7.O_Ready
                feat_op_ready = makespan - operation.release_time
                operation_features_dict['O_Ready'] = feat_op_ready

                normalisation_dict[op_idx] = operation_features_dict

        # scale features values in [0,1]
        features = ['O_MeanPT', 'O_MinPT', 'O_Path_OpNo', 'O_Path_minLen', 'O_Ready']
        self.max_min_scale_of_feature_values(features, normalisation_dict)

        #evaluate tree with scaled values
        for op_idx, redy_op in enumerate(self.operations_redy):
            if redy_op == 1:
                score = 0
                l = []
                op_features = normalisation_dict[op_idx]
                if self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE:
                    for pf in priority_func:
                        _pf = pf[0]
                        a = _pf(op_features['O_MeanPT'], op_features['O_MinPT'],
                                op_features['O_Flex'],
                                op_features['O_Path_OpNo'], op_features['O_Path_minLen'],
                                op_features['O_Ready'])
                        # a = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                        #         feat_op_path_opNO, feat_op_path_minLen, feat_op_ready)
                        l.append(a)
                        score += a
                    score /= len(priority_func)
                elif (self.evaluation_type is GPBase.RuleEvaluationType.BEST or
                      self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE_INSTANCES):
                    _pf = priority_func[0]
                    score = _pf(op_features['O_MeanPT'], op_features['O_MinPT'],
                                op_features['O_Flex'],
                                op_features['O_Path_OpNo'], op_features['O_Path_minLen'],
                                op_features['O_Ready'])

                ready_operations.append((score, op_idx))
        ready_operations.sort(reverse=True)  # higher score first

        _, selected_operation_idx = ready_operations.pop(0)
        selected_operation = self.operations[selected_operation_idx]

        ready_machines = []
        #compute and store features for all available machines
        normalisation_dict = {}
        for m_idx, eligible_machine in enumerate(selected_operation.machines):
            if eligible_machine == 1:
                processing_time = selected_operation.execution_times_setup[m_idx]
                machine_feat_dict = {}

                # Features on operation-machine edges
                # 1. E_PT - Processing time p_{ik}  of operation i on machine k
                feat_edge_processing_time = processing_time
                machine_feat_dict['E_PT'] = feat_edge_processing_time

                # Machines features
                # 1. M_RT - Last operation completion time t_{last}: Determines machine availability.
                feat_machine_ready_time = self.machine_ready_time[m_idx]
                machine_feat_dict['M_RT'] = feat_machine_ready_time

                # 2. M_OP - Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
                feat_machine_operation_proportion = self.machine_operation_no[
                                                        m_idx] / self.no_uncheduled_operations if self.no_uncheduled_operations != 0 else 0
                machine_feat_dict['M_OP'] = feat_machine_operation_proportion

                # 3. M_UT - Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
                feat_machine_utilization_percentage = self.machines[m_idx].get_total_occupancy_duration() / \
                                                      makespan if makespan != 0 else 0
                machine_feat_dict['M_UT'] = feat_machine_utilization_percentage

                # 4. M_QL
                feat_machine_queue_length = self.machine_queue_op_no[m_idx]
                machine_feat_dict['M_QL'] = feat_machine_queue_length

                # 5. M_QD
                feat_machine_queue_duration = self.machine_queue_op_duration[m_idx]
                machine_feat_dict['M_QD'] = feat_machine_queue_duration

                # M_CT_A
                feat_machine_compleation_time_append = max(selected_operation.release_time,
                                                           self.machine_ready_time[m_idx]) + processing_time
                machine_feat_dict['M_CT_A'] = feat_machine_compleation_time_append

                # M_CT_B
                index, start_time, end_time = backward_planning_completion_time(selected_operation,
                                                                                m_idx,
                                                                                self.machines)
                feat_machine_compleation_time_backward = end_time
                machine_feat_dict['M_CT_B'] = feat_machine_compleation_time_backward

                normalisation_dict[m_idx] = machine_feat_dict

        features = ['E_PT', 'M_RT', 'M_QL', 'M_QD', 'M_CT_A', 'M_CT_B']
        self.max_min_scale_of_feature_values(features, normalisation_dict)

        for m_idx, eligible_machine in enumerate(selected_operation.machines):
            if eligible_machine == 1:
                machine_features = normalisation_dict[m_idx]
                score = 0
                l = []
                if self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE:
                    for pf in priority_func:
                        _pf = pf[1]
                        a = _pf(machine_features['E_PT'],#feat_edge_processing_time,
                                machine_features['M_RT'],#feat_machine_ready_time,
                                machine_features['M_OP'],#feat_machine_operation_proportion,
                                machine_features['M_UT'],#feat_machine_utilization_percentage,
                                machine_features['M_QL'],#feat_machine_queue_length,
                                machine_features['M_QD'],#feat_machine_queue_duration,
                                machine_features['M_CT_A'],#feat_machine_compleation_time_append,
                                machine_features['M_CT_B'],#feat_machine_compleation_time_backward
                                )
                        l.append(a)
                        score += a
                    score /= len(priority_func)
                elif (self.evaluation_type is GPBase.RuleEvaluationType.BEST or
                      self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE_INSTANCES):
                    _pf = priority_func[1]
                    score = _pf(machine_features['E_PT'],#feat_edge_processing_time,
                                machine_features['M_RT'],#feat_machine_ready_time,
                                machine_features['M_OP'],#feat_machine_operation_proportion,
                                machine_features['M_UT'],#feat_machine_utilization_percentage,
                                machine_features['M_QL'],#feat_machine_queue_length,
                                machine_features['M_QD'],#feat_machine_queue_duration,
                                machine_features['M_CT_A'],#feat_machine_compleation_time_append,
                                machine_features['M_CT_B'],#feat_machine_compleation_time_backward
                                )

                ready_machines.append((score, m_idx))

        # select pair
        ready_machines.sort(reverse=True)  # higher score first

        _, selected_machine_idx = ready_machines.pop(0)
        self.action_history.append((selected_operation_idx, selected_machine_idx))

        return (selected_operation_idx, selected_machine_idx)

    def get_next_action_1tree(self, priority_func: Any, individual_trees_no=1):
        '''
        Selects the next pair (operation, machine) to be scheduled
        :param priority_func: a GP priority rule or rules that selects a pair (op, machine)
        :return:
        '''
        ready_pairs = []
        max_processing_times_per_machine, no_of_operations_executable_on_machine = self.get_heppler_informations()
        makespan = self.get_makespan()

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

                        # M_CT_A
                        feat_machine_compleation_time_append = max(operation.release_time,
                                                                   self.machine_ready_time[m_idx]) + processing_time
                        # M_CT_B
                        index, start_time, end_time = backward_planning_completion_time(operation,
                                                                                        m_idx,
                                                                                        self.machines)
                        feat_machine_compleation_time_backward = end_time
                        # print("priority_func", priority_func)
                        # print(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                        #                       feat_machine_ready_time, feat_machine_operation_proportion,
                        #                       feat_machine_utilization_percentage,
                        #                       feat_edge_processing_time, feat_edge_PT_maxPT, feat_edge_PT_maxMachinePT)

                        score = 0
                        l=[]
                        if self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE:
                            for pf in priority_func:
                                _pf = pf[0]
                                # a = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                                #           feat_machine_ready_time, feat_machine_operation_proportion,
                                #           feat_machine_utilization_percentage,
                                #           feat_edge_processing_time, feat_edge_PT_maxPT,
                                #            feat_edge_PT_maxMachinePT,
                                #            feat_op_path_opNO, feat_op_path_minLen, feat_op_ready,
                                #            feat_machine_queue_lenght,feat_machine_queue_duration,
                                #            feat_machine_compleation_time_append,
                                #            feat_machine_compleation_time_backward
                                #        )
                                a = _pf(
                                          feat_edge_processing_time, feat_edge_PT_maxPT,
                                          feat_edge_PT_maxMachinePT,
                                           feat_machine_compleation_time_append,
                                           feat_machine_compleation_time_backward
                                       )
                                l.append(a)
                                score+=a
                            score /= len(priority_func)
                        elif self.evaluation_type is  GPBase.RuleEvaluationType.BEST or \
                                self.evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE_INSTANCES:
                            _pf = priority_func[0]
                            # score = _pf(feat_op_mean_time, feat_op_min_time, feat_op_flexibility_factor,
                            #             feat_machine_ready_time, feat_machine_operation_proportion,
                            #             feat_machine_utilization_percentage,
                            #             feat_edge_processing_time, feat_edge_PT_maxPT, feat_edge_PT_maxMachinePT,
                            #             feat_op_path_opNO, feat_op_path_minLen, feat_op_ready,
                            #             feat_machine_queue_lenght,feat_machine_queue_duration,
                            #             feat_machine_compleation_time_append,
                            #             feat_machine_compleation_time_backward
                            #             )
                            score = _pf(feat_edge_processing_time,
                                        feat_edge_PT_maxPT,
                                        feat_edge_PT_maxMachinePT,
                                        feat_machine_compleation_time_append,
                                        feat_machine_compleation_time_backward
                                        )
                        ready_pairs.append((score, op_idx, m_idx))
        # select pair
        ready_pairs.sort(reverse=True)  # higher score first

        _, selected_operation_idx, selected_machine_idx = ready_pairs.pop(0)
        self.action_history.append((selected_operation_idx, selected_machine_idx))

        return (selected_operation_idx, selected_machine_idx)

    def step(self, action, **args):
        '''
        Shedule operation on machine and updates the internal information
        :param action: contains the selected pair (operation, amchine)
        :param args:
        :return:
        '''
        selected_operation_idx = action[0]
        selected_machine_idx = action[1]

        # update operations structure
        selected_operation = self.operations[selected_operation_idx]

        #first interval where it fits
        index_machine_interval, start_time, completion_time = backward_planning_completion_time(selected_operation,
                                                                        selected_machine_idx,
                                                                        self.machines)
        #after last operation scheduled
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
                selected_op_parent.release_time = selected_op_parent.last_child_scheduled_finished

        # update machines
        #self.machines[selected_machine_idx].add_last_interval(selected_operation) - in case of inserting after last operation scheduled
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

        self.done = True if self.operations_redy.count(1) == 0 else False

        return 0, 0, self.done

    def evaluate_instance(self, priority_funcs: Any, get_next_action=1)->float:
            """
            :param priority_func: the function used to select next pair (operation, machine)
            :return:
            """
            self.get_next_instance()

            while not self.done:
                action = self.get_next_action(priority_funcs, get_next_action)
                self.step(action)

            return self.get_makespan() #makespan

    def get_makespan(self, *args):
        return max(self.machine_ready_time)  # makespan

