import json
from statistics import stdev, mean

import numpy as np
import pandas as pd
import itertools

from numpy.ma.extras import average
from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.stats import friedmanchisquare, wilcoxon


train_set_instances = ["50_2_4_12", "50_4_4_51", "50_6_4_91", "50_8_8_154",
"bom_deep_10_5_20_5_No2","bom_deep_6_5_20_10_No1","bom_deep_7_5_10_5_No2","bom_deep_8_3_20_5_No2","bom_deep_9_5_10_5_No1",

"bom_deep_7_3_15_5_No1_p30_ao75_am15_1_m30_ao75_am24_1", "bom_deep_7_3_15_5_No1_p30_ao75_am15_5_m10_ao75_am10_4",
"bom_deep_7_3_15_5_No1_p40_ao81_am15_1_m12_ao81_am12_5", "bom_deep_7_3_15_5_No1_p50_ao87_am15_1_m10_ao87_am10_1",
"bom_deep_7_3_15_5_No1_p50_ao87_am15_1_m25_ao87_am20_3", "bom_deep_7_3_15_5_No1_p50_ao87_am15_2_m25_ao87_am23_5",
"bom_deep_7_3_15_5_No1_p50_ao87_am15_4_m10_ao87_am10_2", "bom_deep_7_3_15_5_No1_p50_ao87_am15_5_m10_ao87_am10_3",

"bom_tubes", "bom_tubes_p10_ao11_am18_3",
"bom_tubes_p50_ao15_am18_3_m10_ao15_am9_2", "bom_tubes_p50_ao15_am18_5_m10_ao15_am10_2",
"bom_tubes_p50_ao15_am18_5_m25_ao15_am17_4", "bom_tubes_p70_ao17_am18_1_m10_ao17_am10_5",
"bom_tubes_p70_ao17_am18_1_m30_ao17_am23_4", "bom_tubes_p80_ao18_am18_2_m13_ao18_am11_5",
"bom_tubes_p80_ao18_am18_5_m35_ao18_am27_1",

"bom_wide_2_10_30_15_No2", "bom_wide_2_11_13_5_No2",
"bom_wide_2_7_20_5_No1", "bom_wide_2_9_10_5_No2",
"bom_wide_3_7_10_5_No1", "bom_wide_4_6_16_4_No2",
"bom_wide_5_10_10_5_No2",

"la03.fjs", "la09.fjs", "la18.fjs", "la19.fjs",
"la24.fjs", "la30.fjs", "la35.fjs"]

datasets={"fjssp-la":["fjssp-la"]*3,
          "wide":["wide"]*3,
          "deep":["deep"]*3,
          "2asp": ["2asp"] * 3,
          "vtubes-test":["vtubes-test","vtubes","vtubes/test"],
          "vtubes-train": ["vtubes-train", "vtubes", "vtubes/train"],
          "vdeep-test": ["vdeep-test", "vdeep", "vdeep/test"],
          "vdeep-train": ["vdeep-train", "vdeep", "vdeep/train"],
          }

dr_algo_cols = ['InstanceUpperBound','fcfs-b', 'fcfs-max', 'fcfs-min',
                 'fop-b', 'fop-max', 'fop-min',
                 'fopno-b-min',	'fopno-max-min', 'fopno-min-min',
                 'letsa-avg', 'letsa-max', 'letsa-min',
                 'mop-b', 'mop-max', 'mop-min',
                 'mopno-b-min',	'mopno-max-min', 'mopno-min-min']#, 'spt-min','spt-b']

def get_gp_data(df, gp_variant, file_path, single_run=True):
    """
    Adds a GP run to a dataframe
    :param df: dataframe where to add the results from current json file
    :param gp_variant: the column name for results (gp variant name)
    :param file_path: the JSON file path
    :param single_run: the JSON files contains a list of algorithm runs
    :return:
    """

    with open(file_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)

    if single_run:
        bom = json_obj["test-data-file-names"]
        ms  = json_obj["agent"]["makespan_tests"]
        apply_gp_time = json_obj["agent"]["running_time"]
        gp_df = pd.DataFrame({"Bom": bom, gp_variant: ms, f'{gp_variant}_runtime': apply_gp_time}).set_index("Bom")

        df = df.join(gp_df, how="left")
    else:
        dfs = []
        for i, json_obj in enumerate(json_obj["runs"]):
            bom = json_obj["test-data-file-names"]
            ms = json_obj["agent"]["makespan_tests"]
            apply_gp_time = json_obj["agent"]["running_time"]

            df_i = pd.DataFrame({
                "Bom": bom,
                f'{gp_variant}': ms,
                "runtime": apply_gp_time,
                "run": i
            })
            dfs.append(df_i)
        gp_all_df = pd.concat(dfs, ignore_index=True)
        #print(gp_all_df)

        gp_min_df = (
            gp_all_df
            .groupby("Bom")
            .agg(
                **{
                    gp_variant: (gp_variant, "min"),
                    f"{gp_variant}_runtime": ("runtime", "sum"),
                    f"mean_{gp_variant}": (gp_variant, "mean"),
                    f"std_{gp_variant}": (gp_variant, "std")
                }
            )
            .reset_index()
            .set_index("Bom")
        )

        #print("gp_min_df", gp_min_df)
        df = df.join(gp_min_df, on="Bom", how="left")

    return df

def load_gp(df, file_path, useRulesAssamble=False):
    """
    Loads a result file and concatenates to an existing dataframe
    :param df: the existing dataframe
    :param file_path: results file path
    :param useRulesAssamble: true for option assamble-instances; false for options: best, assamble
    :return:
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)

    if useRulesAssamble:
        dfs = []
        for i, json_obj in enumerate(json_obj["runs"]):
            bom = json_obj["test-data-file-names"]
            ms = json_obj["agent"]["makespan_tests"]
            apply_gp_time = json_obj["agent"]["running_time"]

            df_i = pd.DataFrame({
                "Bom": bom,
                "Makespan": ms,
                "run": i,
                "RunTime": apply_gp_time
            })
            dfs.append(df_i)
        gp_all_df = pd.concat(dfs, ignore_index=True)
        gp_min_df = (
            gp_all_df
            .groupby("Bom")
            .agg(
                Makespan=("Makespan", "min"),
                RunTime=("RunTime", "sum")
            )
            .reset_index()
        )
        _df = gp_min_df

    else:
        boms = json_obj["test-data-file-names"]
        makespan = json_obj["agent"]["makespan_tests"]
        apply_gp_time = json_obj["agent"]["running_time"]
        _df  = pd.DataFrame({"Bom": boms, "Makespan": makespan, "RunTime" : apply_gp_time})

    if df.empty:
        df = _df
    else:
        df = pd.concat([df, _df],  ignore_index=True)

    return df

def load_model(alg_name, model, path_gp, file_pattern, seeds):
    print("model",model)
    df_ass_inst = pd.DataFrame()
    for seed in seeds:
        df_ass_inst = load_gp(df_ass_inst,
                              f'{path_gp}/{model}/result_{file_pattern}_assemble_inst_seed_{seed}.json', useRulesAssamble=True)
    df_ass_inst["Algorithm"] = f"{alg_name}_ass_inst"

    df_ass = pd.DataFrame()
    for seed in seeds:
        df_ass = load_gp(df_ass,
                         f'{path_gp}/{model}/result_{file_pattern}_assemble_seed_{seed}.json')
    df_ass["Algorithm"] = f"{alg_name}_ass"

    df_best = pd.DataFrame()
    for seed in seeds:
        df_best = load_gp(df_best,
                          f'{path_gp}/{model}/result_{file_pattern}_best_seed_{seed}.json')

    df_best["Algorithm"] = f"{alg_name}_best"
    return df_best, df_ass, df_ass_inst


def sign_test(dataset_name, variant_model, path_drsa, path_gp, seeds, includeTrainInstances=False):
    # load dispatch rules results
    dr_algo_cols = ['InstanceUpperBound', 'fcfs-b', 'fcfs-max', 'fcfs-min',
                 'fop-b', 'fop-max', 'fop-min',
                 'fopno-b-min', 'fopno-max-min', 'fopno-min-min',
                 'letsa-avg', 'letsa-max', 'letsa-min',
                 'mop-b', 'mop-max', 'mop-min',
                 'mopno-b-min', 'mopno-max-min', 'mopno-min-min']

    df_dr = pd.read_csv(f"/{path_drsa}/{dataset_name}-dr.csv")
    df_dr = df_dr.pivot( index="Bom", columns="Algorithm", values="Makespan")
    algo_df = df_dr[dr_algo_cols].replace(-1, np.inf)
    #min makespan value from all DR
    df_dr["dr_min"] = algo_df.min(axis=1)
    # DR names that have makespan = min makespan value from all DR
    df_dr["dr_best_algorithms"] = algo_df.apply(
        lambda row: ", ".join(row.index[row == row.min()]),
        axis=1
    )

    #Simulated annealing
    df_sa = pd.read_csv(f"/{path_drsa}/{dataset_name}-sa-ei-5s-1p.csv")
    df_sa["Algorithm"] = df_sa["Algorithm"].replace('SAL-Ei-LM', 'sa')
    df_sa.rename(columns={"Time": "runtime-sa"}, inplace=True)
    print(df_sa["runtime-sa"])

    common = df_sa

    algs = ["sa"]
    #load gp variants
    for alg_name, model, file_pattern in variant_model.values():
        df_best, df_ass, df_ass_inst = load_model(alg_name, model, path_gp, file_pattern, seeds)
        print(df_best.columns, df_ass.columns, df_ass_inst.columns)
        common = pd.concat([common, df_ass_inst, df_ass, df_best], ignore_index=True)
        algs.extend([f"{alg_name}_ass_inst", f"{alg_name}_ass", f"{alg_name}_best"])

    #################################################################################
    ### Wilcoxon test & GAP relative to best DR
    #################################################################################
    gap_min, gap_min_1000, gap_min_500 = {}, {}, {}
    gap_avg, gap_avg_1000, gap_avg_500 = {}, {}, {}
    for alg in algs:
        gap_min[alg], gap_min_1000[alg], gap_min_500[alg] = [], [], []
        gap_avg[alg], gap_avg_1000[alg], gap_avg_500[alg] = [], [], []

    #build algorithms pairs
    result_wilcoxon = {}
    for alg1 in algs:
        for alg2 in algs:
            if alg1 != alg2:
                result_wilcoxon[f"{alg1} {alg2}"] = {'ties': 0, 'win': 0, 'lose': 0}

    #compute bonferroni constant
    bonferroni_constant = 0.05 / (len(result_wilcoxon)/2)  # Bonferroni constant

    runtime_inst = {alg:[] for alg in algs}

    for instance in common['Bom'].unique():
        #ignore instances that are in the training set
        if (not includeTrainInstances) and instance in train_set_instances:
            continue

        df_instance_filtered = common [common['Bom'] == instance]
        insts_details = {}
        upperBound = df_dr.loc[instance, "dr_min"]

        #construct information related to BOM nodes number that have more than 500, 1000 nodes
        instance_nodes_no = df_instance_filtered['Nodes'].dropna().unique()
        if len(instance_nodes_no) >= 1:
            instance_node_no = instance_nodes_no[0]
        else:
            instance_node_no = 1;

        #agregate information for an instance from multiple runs
        for alg in algs:
            inst_details = {}
            inst_details['runs'] = df_instance_filtered[df_instance_filtered["Algorithm"]==alg]["Makespan"].head(10)
            inst_details['min'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).min()
            inst_details['mean'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).mean()
            inst_details['std'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).std()
            inst_details['runtimes'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["RunTime"].head(10).sum()

            insts_details[alg]=inst_details
            #print(upperBound, inst_details['min'], (upperBound - inst_details['min']) / upperBound)
            gap_min[alg].append((upperBound - inst_details['min']) / upperBound)
            gap_avg[alg].append((upperBound - inst_details['mean']) / upperBound)

            #runtime_inst += inst_details['runtime_sum']
            if alg in runtime_inst:
                    #runtime_inst[alg].extend(inst_details['runtimes'])
                    runtime_inst[alg].append(inst_details['runtimes'])

            if instance_node_no >= 500:
                gap_min_500[alg].append((upperBound - inst_details['min']) / upperBound)
                gap_avg_500[alg].append((upperBound - inst_details['mean']) / upperBound)

            if instance_node_no >= 1000:
                gap_min_1000[alg].append((upperBound - inst_details['min']) / upperBound)
                gap_avg_1000[alg].append((upperBound - inst_details['mean']) / upperBound)

        #construct information for wilcoxon test (win, ties, lose)
        for a, b in itertools.combinations(algs, 2):
            a_details = insts_details[a]
            b_details = insts_details[b]

            d = np.around(np.array(a_details['runs']) - np.array(b_details['runs']), decimals=3)

            if not np.any(d):
                result_wilcoxon[a+" "+b]['ties'] += 1
                result_wilcoxon[b+" "+a]['ties'] += 1
            else:
                r = wilcoxon(d)
                if r.pvalue < bonferroni_constant:
                    if np.array(a_details['runs']).mean() < np.array(b_details['runs']).mean():
                        result_wilcoxon[a+" "+b]['win'] += 1
                        result_wilcoxon[b+" "+a]['lose'] += 1
                    else:
                        result_wilcoxon[a+" "+b]['lose'] += 1
                        result_wilcoxon[b+" "+a]['win'] += 1
                else:
                    result_wilcoxon[a + " "+b]['ties'] += 1
                    result_wilcoxon[b + " "+a]['ties'] += 1

    for k,v in result_wilcoxon.items():
        print(k,";",f"({v['win']},{v['ties']},{v['lose']})")

    ### display GAP relative to best DR
    print("len(common['Bom'].unique())", len(common['Bom'].unique()))
    for alg in algs:
        print(alg, f"gap_min;  {100*average(gap_min[alg]):.2f}")
        print(alg, f"gap_avg;  {100*average(gap_avg[alg]):.2f} \pm {100 * stdev(gap_avg[alg]):.2f}")

    print("inst_no_500", len(gap_min_500['sa']))
    if len(gap_min_500['sa']) > 0:
        for alg in algs:
            print(gap_min_500[alg])
            print(alg, f"gap_min;  {100 * average(gap_min_500[alg]):.2f}")
            print(alg, f"gap_avg;  {100 * average(gap_avg_500[alg]):.2f} \pm {100 * stdev(gap_avg_500[alg]):.2f}")
    else:
        print("No instance with more than 500 nodes")

    print("inst_no_1000", len(gap_min_1000['sa']))
    if len(gap_min_1000['sa']) > 0:
        for alg in algs:
            print(alg, f"gap_min;  {100 * average(gap_min_1000[alg]):.2f}")
            if len(gap_avg_1000['sa']) > 1:
                print(alg, f"gap_avg;  {100 * average(gap_avg_1000[alg]):.2f} \pm {100 * stdev(gap_avg_1000[alg]):.2f}")
            else:
                print(alg, f"gap_avg;  {100 * average(gap_avg_1000[alg]):.2f} \pm 0")
    else:
        print("No instance with more than 500 nodes")

    #################################################################################
    ### Run times
    #################################################################################
    r = common["runtime-sa"].agg(["mean", "std"])
    print("SA runtime;", f"${r['mean']:.2f} \\pm {r['std']:.2f}$")
    print("runtime_inst", runtime_inst.keys())
    for alg in algs:
        if alg in runtime_inst:
            df = pd.DataFrame(runtime_inst[alg], columns=['Runmine'])
            r= df['Runmine'].agg(["mean", "std"])
            print(f"{alg} runtime;", f"${r['mean']:.2f} \\pm {r['std']:.2f}$")


models = {"models-asptrain-large": ("gp_2t", "models-asptrain-large", "gp_2t"),
          # "train_using_scaled_terminals": ("gp_2t_scale", "train_using_scaled_terminals", "gp_2t"),
          # "models-asptrain-large-gp-1t": ("gp_1t", "models-asptrain-large-gp-1t", "gp_pair"),
           "models-asptrain-large-simplified" : ("gp_2t_simplify", "models-asptrain-large-simplified", "gp_dr"),
          }
dataset = datasets['2asp']
# sign_test(dataset[0], models,
#           f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
#           f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
#           [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000])



#######################################################################
##  Parse rules terminals
#######################################################################
def  parse_terminals(terminals, rules, count_all=False):
    for rule in rules:
        for key in terminals:
            occurrences = rule.count(key)
            if occurrences>0:
                if count_all:
                    terminals[key] += occurrences
                else:
                    terminals[key] += 1
    print(len(rules),terminals)
    for terminal, apperances in terminals.items():
        print(terminal, apperances/len(rules)*100)

import matplotlib.pyplot as plt

def grafic_feature_importance(data, out_file_path):
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    #plt.xlabel("Features")
    plt.ylabel("Occurrence no.")
    #plt.title("Feature importance / counts")
    #plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    plt.savefig(out_file_path, dpi=300)
    plt.show()

def call_parse_terminals():
    terminals={"O_MeanPT":0, "O_MinPT":0, "O_Flex":0, "O_Path_OpNo":0, "O_Path_MinLen":0,"O_WT":0}
    operators={'+':0, '-':0,'*':0, "/": 0,"max":0,"min":0, "protected_if": 0}
    with open('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/rules/model-asptrain-large/SelectionRules.txt') as fd:
        rules_optuna_selection = fd.readlines()
        parse_terminals(terminals, rules_optuna_selection)
        grafic_feature_importance(terminals, "opSelectionOnes.png")

        parse_terminals(operators, rules_optuna_selection)
        print("operation selection", operators)

        parse_terminals(terminals, rules_optuna_selection, True)
        grafic_feature_importance(terminals, "opSelection.png")


    terminals = {"E_PT": 0, "M_RT": 0, "M_OP": 0, "M_UT": 0, "M_QL": 0, "M_QD": 0, "M_CT_A":0,"M_CT_B":0}
    operators = {'+': 0, '-': 0, '*': 0, "/": 0, "max": 0, "min": 0, "protected_if": 0}
    with open('/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/rules/model-asptrain-large/DispatchRules.txt', 'r') as fd:
        rules_optuna_dispach = fd.readlines()
    parse_terminals(terminals, rules_optuna_dispach)
    grafic_feature_importance(terminals, "ruteSelectionOnes.png")

    parse_terminals(operators, rules_optuna_dispach)
    print("machine selection", operators)

    parse_terminals(terminals, rules_optuna_dispach, True)
    grafic_feature_importance(terminals, "ruteSelection.png")

#call_parse_terminals()


#######################################################################
### GAP fjssp-la in literature
#######################################################################

def for_fjssp_la(alg_name, variant_model, path_gp, seeds):
    common = pd.DataFrame()

    algs = []
    for alg_name, model, file_pattern in variant_model.values():
        df_best, df_ass, df_ass_inst = load_model(alg_name, model, path_gp, file_pattern, seeds)
        print(df_best.columns, df_ass.columns, df_ass_inst.columns)
        common = pd.concat([common, df_ass_inst, df_ass, df_best], ignore_index=True)
        algs.extend([f"{alg_name}_ass_inst", f"{alg_name}_ass", f"{alg_name}_best"])

    # Results reported in: R. Braune, F. Benda, K. F. Doerner, and R. F. Hartl, “A genetic program-
    # ming learning approach to generate dispatching rules for flexible shop
    # scheduling problems,” International Journal of Production Economics,
    # vol. 243, p. 108342, 2022.
    LB_Braune={
    "la01.fjs":570, "la02.fjs":529, "la03.fjs":477, "la04.fjs":502, "la05.fjs":457,
    "la06.fjs":799, "la07.fjs":749, "la08.fjs":765, "la09.fjs":853, "la10.fjs":804,
    "la11.fjs":1071, "la12.fjs":936, "la13.fjs":1038, "la14.fjs":1070, "la15.fjs":1089,
    "la16.fjs":717, "la17.fjs":646, "la18.fjs":663, "la19.fjs":617, "la20.fjs":756,
    "la21.fjs":800, "la22.fjs":733,  "la23.fjs": 809, "la24.fjs":773, "la25.fjs":751,
    "la26.fjs":1052, "la27.fjs":1084, "la28.fjs":1069, "la29.fjs":993, "la30.fjs":1068,
    "la31.fjs":1520, "la32.fjs":1657, "la33.fjs":1497, "la34.fjs":1535, "la35.fjs":1549,
    "la36.fjs":948, "la37.fjs":986, "la38.fjs":943, "la39.fjs":922, "la40.fjs":955
    }

    #common["LB_Braune"] = common.index.to_series().map(LB_Braune)

    print("column",common)
    print("innstances name", common['Bom'].unique())
    print("algs", algs)

    gaps = {}
    gaps_min ={}

    for instance in common['Bom'].unique():

        df_instance_filtered = common [common['Bom'] == instance]

        #agregate information for an instance from multiple runs
        for alg in algs:
            gap =[]
            for mean_makespan in  df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10):
                gap.append((mean_makespan - LB_Braune[instance]) / LB_Braune[instance] *100)
            gaps[(instance, alg)] = (mean(gap), stdev(gap))

            min_makespan = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).min()
            gaps_min[(instance, alg)] = (min_makespan - LB_Braune[instance]) / LB_Braune[instance] * 100


    for alg in algs:
        print("GAP for best obtained value")
        # display the gap obtain by the rule that generated minimum makespan
        mean1_5 = (gaps_min[('la01.fjs',alg)] +gaps_min[('la02.fjs',alg)]+gaps_min[('la03.fjs',alg)]+gaps_min[('la04.fjs',alg)]+gaps_min[('la05.fjs',alg)] )/ 5
        print(f"strategy {alg} gap: {mean1_5:.2f}")
        mean6_10 = (gaps_min[('la06.fjs', alg)] + gaps_min[('la07.fjs', alg)] + gaps_min[('la08.fjs', alg)] + gaps_min[
            ('la09.fjs', alg)] + gaps_min[('la10.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean6_10:.2f}")
        mean11_15 = (gaps_min[('la11.fjs', alg)] + gaps_min[('la12.fjs', alg)] + gaps_min[('la13.fjs', alg)] + gaps_min[
            ('la14.fjs', alg)] + gaps_min[('la15.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean11_15:.2f}")
        mean16_20 = (gaps_min[('la16.fjs', alg)] + gaps_min[('la17.fjs', alg)] + gaps_min[('la18.fjs', alg)] + gaps_min[
            ('la19.fjs', alg)] + gaps_min[('la20.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean16_20:.2f}")
        mean21_25 = (gaps_min[('la21.fjs', alg)] + gaps_min[('la22.fjs', alg)] + gaps_min[('la23.fjs', alg)] + gaps_min[
            ('la24.fjs', alg)] + gaps_min[('la25.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean21_25:.2f}")
        mean26_30 = (gaps_min[('la26.fjs', alg)] + gaps_min[('la27.fjs', alg)] + gaps_min[('la28.fjs', alg)] + gaps_min[
            ('la29.fjs', alg)] + gaps_min[('la30.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean26_30:.2f}")
        mean31_35 = (gaps_min[('la31.fjs', alg)] + gaps_min[('la32.fjs', alg)] + gaps_min[('la33.fjs', alg)] + gaps_min[
            ('la34.fjs', alg)] + gaps_min[('la35.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean31_35:.2f}")
        mean36_40 = (gaps_min[('la36.fjs', alg)] + gaps_min[('la37.fjs', alg)] + gaps_min[('la38.fjs', alg)] + gaps_min[
            ('la39.fjs', alg)] + gaps_min[('la40.fjs', alg)]) / 5
        print(f"strategy {alg} gap: {mean36_40:.2f}")


        # display mean gap and stdev
        print("mean GAP on all rules")
        mean1_5 = (gaps[('la01.fjs', alg)][0] + gaps[('la02.fjs', alg)][0] + gaps[('la03.fjs', alg)][0] + gaps[
            ('la04.fjs', alg)][0] + gaps[('la05.fjs', alg)][0]) / 5
        stdev1_5 = (gaps[('la01.fjs', alg)][1] + gaps[('la02.fjs', alg)][1] + gaps[('la03.fjs', alg)][1] + gaps[
            ('la04.fjs', alg)][1] + gaps[('la05.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean1_5:.2f} \pm {stdev1_5:.2f}" )

        mean6_10 = (gaps[('la06.fjs', alg)][0] + gaps[('la07.fjs', alg)][0] + gaps[('la08.fjs', alg)][0] + gaps[
            ('la09.fjs', alg)][0] + gaps[('la10.fjs', alg)][0]) / 5
        stdev6_10 = (gaps[('la06.fjs', alg)][1] + gaps[('la07.fjs', alg)][1] + gaps[('la08.fjs', alg)][1] + gaps[
            ('la09.fjs', alg)][1] + gaps[('la10.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean6_10:.2f} \pm {stdev6_10:.2f}" )

        mean11_15 = (gaps[('la11.fjs', alg)][0] + gaps[('la12.fjs', alg)][0] + gaps[('la13.fjs', alg)][0] + gaps[
            ('la14.fjs', alg)][0] + gaps[('la15.fjs', alg)][0]) / 5
        stdev11_15 = (gaps[('la11.fjs', alg)][1] + gaps[('la12.fjs', alg)][1] + gaps[('la13.fjs', alg)][1] + gaps[
            ('la14.fjs', alg)][1] + gaps[('la15.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean11_15:.2f} \pm {stdev11_15:.2f}" )

        mean16_20 = (gaps[('la16.fjs', alg)][0] + gaps[('la17.fjs', alg)][0] + gaps[('la18.fjs', alg)][0] + gaps[
            ('la19.fjs', alg)][0] + gaps[('la20.fjs', alg)][0]) / 5
        stdev16_20 = (gaps[('la16.fjs', alg)][1] + gaps[('la17.fjs', alg)][1] + gaps[('la18.fjs', alg)][1] + gaps[
            ('la19.fjs', alg)][1] + gaps[('la20.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean16_20:.2f} \pm {stdev16_20:.2f}" )

        mean21_25 = (gaps[('la21.fjs', alg)][0] + gaps[('la22.fjs', alg)][0] + gaps[('la23.fjs', alg)][0] + gaps[
            ('la24.fjs', alg)][0] + gaps[('la25.fjs', alg)][0]) / 5
        stdev21_25 = (gaps[('la21.fjs', alg)][1] + gaps[('la22.fjs', alg)][1] + gaps[('la23.fjs', alg)][1] + gaps[
            ('la24.fjs', alg)][1] + gaps[('la25.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean21_25:.2f} \pm {stdev21_25:.2f}")

        mean26_30 = (gaps[('la26.fjs', alg)][0] + gaps[('la27.fjs', alg)][0] + gaps[('la28.fjs', alg)][0] + gaps[
            ('la29.fjs', alg)][0] + gaps[('la30.fjs', alg)][0]) / 5
        stdev26_30 = (gaps[('la26.fjs', alg)][1] + gaps[('la27.fjs', alg)][1] + gaps[('la28.fjs', alg)][1] + gaps[
            ('la29.fjs', alg)][1] + gaps[('la30.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean26_30:.2f} \pm {stdev26_30:.2f}" )

        mean31_35 = (gaps[('la31.fjs', alg)][0] + gaps[('la32.fjs', alg)][0] + gaps[('la33.fjs', alg)][0] + gaps[
            ('la34.fjs', alg)][0] + gaps[('la35.fjs', alg)][0]) / 5
        stdev31_35 = (gaps[('la31.fjs', alg)][1] + gaps[('la32.fjs', alg)][1] + gaps[('la33.fjs', alg)][1] + gaps[
            ('la34.fjs', alg)][1] + gaps[('la35.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean31_35:.2f} \pm {stdev31_35:.2f}" )

        mean36_40 = (gaps[('la36.fjs', alg)][0] + gaps[('la37.fjs', alg)][0] + gaps[('la38.fjs', alg)][0] + gaps[
            ('la39.fjs', alg)][0] + gaps[('la40.fjs', alg)][0]) / 5
        stdev36_40 = (gaps[('la36.fjs', alg)][1] + gaps[('la37.fjs', alg)][1] + gaps[('la38.fjs', alg)][1] + gaps[
            ('la39.fjs', alg)][1] + gaps[('la40.fjs', alg)][1]) / 5
        print(f"strategy {alg} mean-gap: {mean36_40:.2f} \pm {stdev36_40:.2f}" )


models = {"models-asptrain-large": ("gp_2t", "models-asptrain-large", "gp_2t"),
          # "train_using_scaled_terminals": ("gp_2t_scale", "train_using_scaled_terminals", "gp_2t"),
          # "models-asptrain-large-gp-1t": ("gp_1t", "models-asptrain-large-gp-1t", "gp_pair"),
          #"models-asptrain-large-simplified" : ("gp_2t_simplify", "models-asptrain-large-simplified", "gp_dr"),
          }
dataset = datasets['fjssp-la']
for_fjssp_la(dataset[0], models,
             f"../../results/{dataset[1]}",
              [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000])
