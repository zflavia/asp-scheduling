from src.agents.gp.gp_1tree import GP_One_Tree
from src.agents.gp.gp_2trees import GP_Two_Trees
from src.agents.gp.simpleTree import  simplify_individual, infix_str, tree_str
from numbers import Number
from  statistics import mean
import pickle


def count_unique_terminal_used(tree):
    unique_terminal_set = set()
    for node in tree:
        if node.arity == 0 and hasattr(node, "value") and (not isinstance(node.value, Number)):
            unique_terminal_set.add(node.value)
    return len(unique_terminal_set)


def get_tree_statistics_1tree(model_path, model_name, instances):
    file = "/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/models/gp/models-asptrain-large-gp5t/gp_pair_5t_ASPTrain_optunaParam_seed_2500"
    stat_not_simplified = {"nodes_no" : [], "depth" : [], "unique_terminals" : [], "terminals_used": []}
    stat_simplified = {"nodes_no" : [], "depth" : [], "unique_terminals" : [], "terminals_used": []}

    for instance in instances:
        path = f"{model_path}/{model_name}{instance}.pkl"
        with open(path, "rb") as handle:
            data = pickle.load(handle)
            for el in data['hof']:
                print("~~~~~", el)
                print(f"  Initial   : {infix_str(el)}")

                simp = simplify_individual(el, GP_One_Tree.configure_terminals())
                print(f"  Simplified: {infix_str(simp)}")
                print("~~~~~", simp)

                stat_not_simplified["nodes_no"].append(len(el))
                stat_not_simplified["depth"].append(el.height)
                stat_not_simplified["unique_terminals"].append(count_unique_terminal_used(el))
                stat_not_simplified["terminals_used"].append(count_unique_terminal_used(el)/5)

                stat_simplified["nodes_no"].append(len(simp))
                stat_simplified["depth"].append(simp.height)
                stat_simplified["unique_terminals"].append(count_unique_terminal_used(simp))
                stat_simplified["terminals_used"].append(count_unique_terminal_used(simp)/5)

    rules_no = (len(stat_simplified["nodes_no"]))

    print("Measure", "Before Simplification", "After Simplification", "reduction")
    print("Nodes no", mean(stat_not_simplified["nodes_no"]), mean(stat_simplified["nodes_no"]),
          100-mean(stat_simplified["nodes_no"])/mean(stat_not_simplified["nodes_no"])*100)
    print("Expr depth", mean(stat_not_simplified["depth"]), mean(stat_simplified["depth"]),
          100-mean(stat_simplified["depth"])/ mean(stat_not_simplified["depth"]) * 100)
    print("unique_terminals", mean(stat_not_simplified["unique_terminals"]), mean(stat_simplified["unique_terminals"]),
          100-mean(stat_simplified["unique_terminals"])/mean(stat_not_simplified["unique_terminals"]) * 100)
    print("terminals used", mean(stat_not_simplified["terminals_used"]), mean(stat_simplified["terminals_used"]),
          100-mean(stat_simplified["terminals_used"])/mean(stat_not_simplified["terminals_used"]) * 100)


def get_tree_statistics_2trees(model_path, model_name, instances):
    stat_not_simplified = {"nodes_no": [], "depth": [], "unique_terminals": [], "terminals_used": []}
    stat_simplified = {"nodes_no": [], "depth": [], "unique_terminals": [], "terminals_used": []}

    pset_dispatch, pset_machines = GP_Two_Trees.configure_terminals()
    for instance in instances:
        path = f"{model_path}/{model_name}{instance}.pkl"
        with open(path, "rb") as handle:
            data = pickle.load(handle)
            for tree in data['hof']:
                el = tree[0]
                #el = simplify_individual(el[1], GP_Two_Trees.pset_route)

                #print("~~~~~", el)
                print(f"  Initial   : {infix_str(el)}")

                simp = simplify_individual(el, pset_dispatch)#pset_dispatch, pset_machines)
                print(f"  Simplified: {infix_str(simp)}")
                print("~~~~~   el:", el)
                print("~~~~~ simp:", simp)

                stat_not_simplified["nodes_no"].append(len(el))
                stat_not_simplified["depth"].append(el.height)
                stat_not_simplified["unique_terminals"].append(count_unique_terminal_used(el))
                stat_not_simplified["terminals_used"].append(count_unique_terminal_used(el) / 6)

                stat_simplified["nodes_no"].append(len(simp))
                stat_simplified["depth"].append(simp.height)
                stat_simplified["unique_terminals"].append(count_unique_terminal_used(simp))
                stat_simplified["terminals_used"].append(count_unique_terminal_used(simp) / 6)


    rules_no = (len(stat_simplified["nodes_no"]))


    print("Measure", "Before Simplification", "After Simplification", "reduction")
    print("Nodes no", mean(stat_not_simplified["nodes_no"]), mean(stat_simplified["nodes_no"]),
          100-mean(stat_simplified["nodes_no"])/mean(stat_not_simplified["nodes_no"])*100)
    print("Expr depth", mean(stat_not_simplified["depth"]), mean(stat_simplified["depth"]),
          100-mean(stat_simplified["depth"])/ mean(stat_not_simplified["depth"]) * 100)
    print("unique_terminals", mean(stat_not_simplified["unique_terminals"]), mean(stat_simplified["unique_terminals"]),
          100-mean(stat_simplified["unique_terminals"])/mean(stat_not_simplified["unique_terminals"]) * 100)
    print("ns", stat_not_simplified["depth"], "\ns ", stat_simplified["depth"])
    print("terminals used", mean(stat_not_simplified["terminals_used"]), mean(stat_simplified["terminals_used"]),
          100 - mean(stat_simplified["terminals_used"]) / mean(stat_not_simplified["terminals_used"]) * 100)


get_tree_statistics_1tree("/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/models/gp/models-asptrain-large-gp5t/",
                          "gp_pair_5t_ASPTrain_optunaParam_seed_",
                          [0,200,400,600,800,1000,1500,2000,2500,3000])

# get_tree_statistics_2trees("/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/data/models/gp/models-asptrain-large/",
#                           "gp_dr_ASPTrain_optunaParam_seed_",
#                           [0,200,400,600,800,1000,1500,2000,2500,3000])



