import pandas as pd
import pickle
data = pd.read_pickle('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/instances/asp/config_ASP_TUBES_ORIGINAL_GNN_train-flavia.pkl')
print(len(data))

for instance in data[:2]:
    for task in instance:
        print("Task index", task.task_index, "quatity", task.quantity, task.machines)
        # for machine_id in task.execution_times:
        #     print("\tmachine", machine_id, "setup", task.setup_times[machine_id], "exectime", task.execution_times[machine_id], "execution_times_setup", task.execution_times_setup[machine_id]  )
