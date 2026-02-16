import json
import glob

terminals_operations = ["O_MeanPT", "O_MinPT", "O_Flex", "O_Path_OpNo", "O_Path_MinLen", "O_WT"]
terminals_machines = ["E_PT", "M_RT", "M_OP", "M_UT", "M_QL", "M_QD", "M_CT_A", "M_CT_B"]

operators = ['+', '-', '*', '/', 'max', 'min', 'protected_if', 'lt']


def count_terminal(rule_list: list[str], terminals: list[str]) :
    all_occurrences = {}
    occurrences = {}
    for terminal in terminals:
        occurrences[terminal] = 0
        all_occurrences[terminal] = 0

    for rule in rule_list:
           for key in terminals:
               c = rule.count(key)
               if c > 0:
                    occurrences[key] += 1
                    all_occurrences[key] += c
    return occurrences, all_occurrences

def extract_operation_machines_rules(path):
    # absolute path to search all text files inside a specific folder
    path = rf'{path}/*.txt'
    files_path = glob.glob(path, recursive=True)
    selection_rules_operation = []
    selection_rules_machine = []

    for file_path in files_path:
        with open(file_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)
            hof_individuals = json_obj['hof']
            for individual in hof_individuals:
                selection_rules_operation.append(individual[0])
                selection_rules_machine.append(individual[1])

    occurrences, all_occurrences = count_terminal(selection_rules_operation, terminals_operations)
    print("Operations",'\n\t', occurrences, '\n\t', all_occurrences)
    occurrences, all_occurrences = count_terminal(selection_rules_machine, terminals_machines)
    print("Machines", '\n\t', occurrences, '\n\t', all_occurrences)

    occurrences, all_occurrences = count_terminal(selection_rules_operation, operators)
    print("Operations Operators", '\n\t', occurrences, '\n\t', all_occurrences)
    occurrences, all_occurrences = count_terminal(selection_rules_machine, operators)
    print("Machine Operators", '\n\t', occurrences, '\n\t', all_occurrences)



# print('train-gp-train')
# extract_operation_machines_rules('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/models/gp/models-asptrain-large')
#
# print('train-la')
# extract_operation_machines_rules('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/models/gp/train-on-datasets/fjssp-la')

print('train_using_scaled_terminals')
extract_operation_machines_rules('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/models/gp/train_using_scaled_terminals')
