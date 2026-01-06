from src.data_generator.task import Task
from typing import List
from src.models.machine import Machine

def backward_planning_completion_time(operation: Task,
                                      m_idx: int,
                                      machines: List[Machine]):
    """
    This function performs the logic, with which the machine is chosen using the completion time computed
    from critical path (in the case of the ASP using LETSA heuristic whose steps e the described in the original paper)

    :param task: Task

    :return: Machine on which the task will be scheduled.
    machine_id:  id of selected machine
    latest_start_time:  starting time for task scheduled on selected machine
    end_time:  end time for task scheduled on selected machine
    index:   index in the list of time intervals/machine

    """

    # 4.5.1 Identify the latest available starting time  for operation Je (to verify constraint (2.6) of (PI».
    # 4.5.2 If latest available starting time Sc = Cc - tc, such that the machine is
    # available during (Sc, Cc); Sc, Cc are ideal starting and completion times,
    # respectively. Else select max{Se} as the latest available starting time such that



    duration = operation.execution_times_setup[m_idx] #duration of task exec on mach
    start =  operation.last_child_scheduled_finished
    intrevals_no = machines[m_idx].get_int_len()


    # No intervals scheduled on current machine
    if intrevals_no == 0:
        return (0, start, start + duration)

    else:
        intervals = [[task.started,task.finished] for task in machines[m_idx].intervals]
        intervals.sort(key=lambda x: x[0])

        s = start
        e = s + duration

        for i, (a, b) in enumerate(intervals):
            # Dacă încape înainte de intervalul curent [a,b)
            if e <= a:
                # print("insert i", i, operation.last_child_scheduled_finished, "duration", duration, "s,e",s,e)
                # print(intervals, )
                return (i, s, e)

            # Dacă se suprapune sau cade în interior, îl împingem după acest interval
            if s < b:
                s = b
                e = s + duration

        # Dacă nu a încăput nicăieri, îl punem după ultimul
        return (intrevals_no, s, e)

