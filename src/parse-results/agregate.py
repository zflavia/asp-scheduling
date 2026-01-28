import json
import csv
import numpy as np
import pandas as pd
import itertools
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

dr_algo_cols = ['InstanceUpperBound',	'fcfs-b',	'fcfs-max',	'fcfs-min',
                 'fop-b',	'fop-max',	'fop-min',
                 'fopno-b-min',	'fopno-max-min',	'fopno-min-min',
                 'letsa-avg',	'letsa-max',	'letsa-min',
                 'mop-b',	'mop-max',	'mop-min',
                 'mopno-b-min',	'mopno-max-min',	'mopno-min-min']#, 'spt-min','spt-b']

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


def load_dataset(dataset_name, model_dr, model_pair, path_drsa, path_gp, seeds, out_fine_name, load_gp_pair=False):
    """

    :param dataset_name:
    :param model:
    :param path_drsa:
    :param path_gp:
    :param seeds:
    :return:
    """
    df_sa = pd.read_csv(f"/{path_drsa}/{dataset_name}-sa-ei.csv")
    df_sa = ( df_sa
        .groupby(["Algorithm", "Bom", "Nodes"])
        .agg(
            makespan_mean=("Makespan", "mean"),
            makespan_std=("Makespan", "std"),
            runs=("Makespan", "count")
        )
        .reset_index()
    ).set_index("Bom")

    #dispach rules
    df = pd.read_csv(f"/{path_drsa}/{dataset_name}-dr.csv")

    df_dr = df.pivot(
        index="Bom",
        columns="Algorithm",
        values="Makespan"
    )
    df =  pd.concat([df_sa, df_dr], axis=1)

    #GP
    for seed in seeds:
        df = get_gp_data(df, f'GP_dr_ass_inst_{seed}',
                               f'{path_gp}/{model_dr}/result_gp_dr_assemble_inst_seed_{seed}.json',
                         single_run=False)
    for seed in seeds:
        df = get_gp_data(df, f'GP_dr_ass_{seed}',
                               f'{path_gp}/{model_dr}/result_gp_dr_assemble_seed_{seed}.json')

    for seed in seeds:
        df = get_gp_data(df, f'GP_dr_best_{seed}',
                               f'{path_gp}/{model_dr}/result_gp_dr_best_seed_{seed}.json')

    if load_gp_pair:
        for seed in seeds:
            df = get_gp_data(df, f'GP_pair_ass_inst_{seed}',
                                   f'{path_gp}/{model_pair}/result_gp_pair_assemble_inst_seed_{seed}.json',
                             single_run=False)
        for seed in seeds:
            df = get_gp_data(df, f'GP_pair_ass_{seed}',
                                  f'{path_gp}/{model_pair}/result_gp_pair_assemble_seed_{seed}.json')

        for seed in seeds:
            df = get_gp_data(df, f'GP_pair_best_{seed}',
                              f'{path_gp}/{model_pair}/result_gp_pair_best_seed_{seed}.json')

    #care algoritmi ating minimul
    df = df.rename(columns={"makespan_mean": "SA"})
    columns_names = df.columns
    meta_cols = ["Bom", "Algorithm", "Nodes", "makespan_std", "runs"]
    meta_cols.extend([column for column in columns_names if column.startswith("std_")])
    meta_cols.extend([column for column in columns_names if column.startswith("mean_")])
    meta_cols.extend([column for column in columns_names if column.endswith("_runtime")])
    print(meta_cols)
    algo_cols = [c for c in df.columns if c not in meta_cols]
    algo_df = df[algo_cols].replace(-1, np.inf)
    df["min_makespan"] = algo_df.min(axis=1)
    df["best_algorithms"] = algo_df.apply(
        lambda row: ", ".join(row.index[row == row.min()]),
        axis=1
    )


    algo_df = df[dr_algo_cols].replace(-1, np.inf)
    df["dr_min"] = algo_df.min(axis=1)
    df["dr_best_algorithms"] = algo_df.apply(
        lambda row: ", ".join(row.index[row == row.min()]),
        axis=1
    )

    #pe familie determin minimul
    ass_inst_cols = [c for c in df.columns if c.startswith("GP_dr_ass_inst") and not c.endswith("_runtime")]
    ass_cols = [c for c in df.columns if (c.startswith("GP_dr_ass") and not c.startswith("GP_dr_ass_inst")) and not c.endswith("_runtime")]
    best_cols = [c for c in df.columns if c.startswith("GP_dr_best") and not c.endswith("_runtime")]
    ass_inst_df =df[ass_inst_cols]
    ass_df =df[ass_cols]
    best_df= df[best_cols]

    df["gp_dr_ass_inst"] = ass_inst_df.min(axis=1)
    df["gp_dr_ass"] = ass_df.min(axis=1)
    df["gp_dr_best"] = best_df.min(axis=1)


    if load_gp_pair:
        ass_inst_cols = [c for c in df.columns if c.startswith("GP_pair_ass_inst") and not c.endswith("runtime")]
        ass_cols = [c for c in df.columns if (c.startswith("GP_pair_ass") and not c.startswith("GP_pair_ass_inst")) and not c.endswith("runtime")]
        best_cols = [c for c in df.columns if c.startswith("GP_pair_best") and not c.endswith("runtime")]
        ass_inst_df = df[ass_inst_cols]
        ass_df = df[ass_cols]
        best_df = df[best_cols]

        df["gp_pair_ass_inst"] = ass_inst_df.min(axis=1)
        df["gp_pair_ass"] = ass_df.min(axis=1)
        df["gp_pair_best"] = best_df.min(axis=1)

    gp_min_cols = [c for c in df.columns if (c.startswith("GP_pair") or c.startswith("GP_dr")) and not c.endswith("runtime")]
    gp_min_df = df[gp_min_cols]
    df["gp_min"] = gp_min_df.min(axis=1)

    # pe familie determin runtime
    ass_inst_cols_runtime = [c for c in df.columns if c.startswith("GP_dr_ass_inst") and c.endswith("runtime")]
    ass_cols_runtime = [c for c in df.columns if
                        (c.startswith("GP_dr_ass") and not c.startswith("GP_dr_ass_inst")) and c.endswith("runtime")]
    best_cols_runtime = [c for c in df.columns if c.startswith("GP_dr_best") and c.endswith("runtime")]
    ass_inst_df_runtime = df[ass_inst_cols_runtime]
    ass_df_runtime = df[ass_cols_runtime]
    best_df_runtime = df[best_cols_runtime]

    df["gp_dr_ass_inst_runtime"] = ass_inst_df_runtime.sum(axis=1)
    df["gp_dr_ass_runtime"] = ass_df_runtime.sum(axis=1)
    df["gp_dr_best_runtime"] = best_df_runtime.sum(axis=1)
    df["gp_dr_total_runtime"] = df["gp_dr_ass_inst_runtime"] + df["gp_dr_ass_runtime"] + df["gp_dr_best_runtime"]


    if load_gp_pair:
        ass_inst_cols_runtime = [c for c in df.columns if c.startswith("GP_pair_ass_inst") and c.endswith("runtime")]
        ass_cols_runtime = [c for c in df.columns if
                            (c.startswith("GP_pair_ass") and not c.startswith("GP_pair_ass_inst")) and c.endswith(
                                "runtime")]
        best_cols_runtime = [c for c in df.columns if c.startswith("GP_pair_best") and c.endswith("runtime")]
        ass_inst_df_runtime = df[ass_inst_cols_runtime]
        ass_df_runtime = df[ass_cols_runtime]
        best_df_runtime = df[best_cols_runtime]

        df["gp_pair_ass_inst_runtime"] = ass_inst_df_runtime.sum(axis=1)
        df["gp_pair_ass_runtime"] = ass_df_runtime.sum(axis=1)
        df["gp_pair_best_runtime"] = best_df_runtime.sum(axis=1)
        df["gp_pair_total_runtime"] = df["gp_pair_ass_inst_runtime"] + df["gp_pair_ass_runtime"] + df["gp_pair_best_runtime"]

        print("rt gp_pair_ass_inst_runtime", f"${df['gp_pair_ass_inst_runtime'].mean():.3f} \\pm {df['gp_pair_ass_inst_runtime'].std():.3f}$")
        print("rt gp_pair_ass_runtime", f"${df['gp_pair_ass_runtime'].mean():.3f} \\pm {df['gp_pair_ass_runtime'].std():.3f}$")
        print("rt gp_pair_best_runtime", f"${df['gp_pair_best_runtime'].mean():.3f} \\pm {df['gp_pair_best_runtime'].std():.3f}$")

    std_cols_ = [c for c in df.columns if c.startswith("std")]
    std_df = df[std_cols_]
    df["std=0"] = std_df.isin([0]).sum(axis=1)

    print("HOF:",(df["std=0"].mean()/10)*100)

    print("rt gp_dr_best_runtime", f"${df['gp_dr_best_runtime'].mean():.3f} \\pm {df['gp_dr_best_runtime'].std():.2f}$")


    strategies = ["gp_dr_ass_inst", "gp_dr_ass",  "gp_dr_best", "SA"]
    if load_gp_pair:
        strategies.extend([ "gp_pair_ass_inst", "gp_pair_ass", "gp_pair_best"])

    #comparație pairwise (toți contra tuturor)
    print(strategies)

    for a, b in itertools.combinations(strategies, 2):
        wins_a = (df[a] < df[b]).sum()
        wins_b = (df[b] < df[a]).sum()
        ties = (df[a] == df[b]).sum()

        print(f"{a} vs. {b}; {wins_a};{wins_b};{ties}")

    #rank mediu
    ranks = df[strategies].rank(axis=1, method="average")
    mean_ranks = ranks.mean()
    print(";".join(f"{k};{v:.3f}\n" for k, v in mean_ranks.items()))

    #Friedman test
    if load_gp_pair:
        stat, p = friedmanchisquare(
            df["gp_dr_ass_inst"],
            df["gp_dr_ass"],
            df["gp_dr_best"],
            df["gp_pair_ass_inst"],
            df["gp_pair_ass"],
            df["gp_pair_best"],
            df["SA"]
        )
    else:
        stat, p = friedmanchisquare(
            df["gp_dr_ass_inst"],
            df["gp_dr_ass"],
            df["gp_dr_best"],
            df["SA"]
        )

    print("Friedman p-value:", p)

    strategies = ["gp_min", "SA", "dr_min"]
    results = {}

    for a, b in itertools.combinations(strategies, 2):
        wins_a = (df[a] < df[b]).sum()
        wins_b = (df[b] < df[a]).sum()
        ties = (df[a] == df[b]).sum()

        print(f"{a} vs. {b}; {wins_a};{wins_b};{ties}")


    # rank mediu
    ranks = df[strategies].rank(axis=1, method="average")
    mean_ranks = ranks.mean().sort_values()
    print(";".join(f"{k};{v:.3f}\n" for k, v in mean_ranks.items()))

    stat, p = friedmanchisquare(
        df['dr_min'],
        df["gp_min"],
        df["SA"]
    )
    print("Friedman p-value:", p)

    r = df["gp_dr_ass_inst_runtime"].agg(["mean", "std"])
    print("m ass_inst_df_runtime;",f"${r['mean']:.2f} \\pm {r['std']:.2f}$" )

    r = df["gp_dr_ass_runtime"].agg(["mean", "std"])
    print("m gp_dr_ass_runtime;", f"${r['mean']:.2f} \\pm {r['std']:.2f}$")

    r = df["gp_dr_best_runtime"].agg(["mean", "std"])
    print("m gp_dr_best_runtime;", f"${r['mean']:.3f} \\pm {r['std']:.2f}$")

    if load_gp_pair:
        r = df["gp_pair_ass_inst_runtime"].agg(["mean", "std"])
        print("m gp_pair_ass_inst_runtime;", f"${r['mean']:.2f} \\pm {r['std']:.2f}$")

        r = df["gp_pair_ass_runtime"].agg(["mean", "std"])
        print("m gp_pair_ass_runtime;", f"${r['mean']:.2f} \\pm {r['std']:.2f}$")

        r = df["gp_pair_best_runtime"].agg(["mean", "std"])
        print("m gp_pair_best_runtime;", f"${r['mean']:.3f} \\pm {r['std']:.2f}$")


    df.to_excel(f"{out_fine_name}_{dataset_name}_{model_dr}_{model_pair}.xlsx")

    return df

def generate_data_set_results():
    dataset = datasets["2asp"]
    # load_dataset(dataset[0],"models-asptrain-large", None,
    #              f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
    #              f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
    #              [0,200,400,600,800,1000,1500,2000,2500,3000],
    #              "results_time_optuna")
    load_dataset(dataset[0], "models-asptrain-large", "models-asptrain-large-gp5t",
                 f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
                 f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
                 [0, 200],
                 "results",
                 True)

#generate_data_set_results()



def count_dispach_rules_wins():
    count_dr = {}
    count_no_letsa ={}
    for alg in dr_algo_cols:
        count_dr[alg] = 0
        count_no_letsa[alg]=0
#best_algorithms
    for key,dataset in datasets.items():
        #print(dataset)
        df = load_dataset(dataset[0], "models-asptrain-large", None,
                     f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
                     f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
                     [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000],
                     "results_time_optuna")
        #print(df['dr_best_algorithms'])
        for val in df['dr_best_algorithms']:
            no_letsa = False
            if 'letsa' not in val:
                no_letsa = True
            algs = val.split(',')
            for alg in algs:
                count_dr[alg.strip()] += 1
                if no_letsa:
                    count_no_letsa[alg.strip()] += 1

    for k, v in sorted(count_dr.items(), key=lambda item: item[1], reverse=True):
        print(k, v)

    print("-------no_letsa")
    for k, v in sorted(count_no_letsa.items(), key=lambda item: item[1], reverse=True):
        print(k, v)

#count_dispach_rules_wins()

def count_gp_rules(seeds):
     #ma uit pe cele castigatoare
     count_gp={}
     for seed in seeds:
         count_gp[f'GP_dr_ass_inst_{seed}'] = 0
         count_gp[f'GP_dr_ass_{seed}'] = 0
         count_gp[f'GP_dr_best_{seed}'] = 0

     for key, dataset in datasets.items():
         # print(dataset)
         df = load_dataset(dataset[0], "models-asptrain-large", None,
                           f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
                           f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
                           [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000],
                           "results_time_optuna")

         for val in df['best_algorithms']:
             algs = val.split(',')
             for alg in algs:
                 if alg.strip() in count_gp.keys():
                     count_gp[alg.strip()] += 1

     for k, v in sorted(count_gp.items(), key=lambda item: item[1], reverse=True):
        print(k, v)

     #fff
     rules = {}
     for key, dataset in datasets.items():
         for seed in seeds:
             file_path = f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}/models-asptrain-large/result_gp_dr_assemble_inst_seed_{seed}.json"
             with open(file_path, "r", encoding="utf-8") as f:
                json_obj = json.load(f)

             for i, json_obj in enumerate(json_obj["runs"]):
                mean_ms = json_obj["agent"]["makespan_mean"]
                key = f"rule_{seed}_{i}"
                if key not in rules.keys():
                     rules[key] = 0
                rules[key] += mean_ms

     print("---------Rules------------")
     for k, v in sorted(rules.items(), key=lambda item: item[1], reverse=False):
        print(k, v)

def for_fjssp_la():
    dataset = datasets['fjssp-la']
    seeds = [0,200,400,600,800,1000,1500,2000,2500,3000]
    df = load_dataset(dataset[0], "models-asptrain-large", "models-asptrain-largel-gp5t",
                      f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
                      f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
                      seeds,
                      "results_time_optuna____",)

    # count_gp[f'GP_dr_ass_inst_{seed}'] = 0
    # count_gp[f'GP_dr_ass_{seed}'] = 0
    # count_gp[f'GP_dr_best_{seed}'] = 0

    LB_Braune={
    "la01.fjs":570,
    "la02.fjs":529,
    "la03.fjs":477,
    "la04.fjs":502,
    "la05.fjs":457,
    "la06.fjs":799,
    "la07.fjs":749,
    "la08.fjs":765,
    "la09.fjs":853,
    "la10.fjs":804,
    "la11.fjs":1071,
    "la12.fjs":936,
    "la13.fjs":1038,
    "la14.fjs":1070,
    "la15.fjs":1089,
    "la16.fjs":717,
    "la17.fjs":646,
    "la18.fjs":663,
    "la19.fjs":617,
    "la20.fjs":756,
    "la21.fjs":800,
    "la22.fjs":733,
    "la23.fjs": 809,
    "la24.fjs":773,
    "la25.fjs":751,
    "la26.fjs":1052,
    "la27.fjs":1084,
    "la28.fjs":1069,
    "la29.fjs":993,
    "la30.fjs":1068,
    "la31.fjs":1520,
    "la32.fjs":1657,
    "la33.fjs":1497,
    "la34.fjs":1535,
    "la35.fjs":1549,
    "la36.fjs":948,
    "la37.fjs":986,
    "la38.fjs":943,
    "la39.fjs":922,
    "la40.fjs":955}


    df["LB_Braune"] = df.index.to_series().map(LB_Braune)

    for seed in seeds:
        df[f"GAP_dr_ass_inst_{seed}"] = (df[f"GP_dr_ass_inst_{seed}"] - df["LB_Braune"]) / df["LB_Braune"]*100
        df[f"GAP_dr_ass_{seed}"] = (df[f"GP_dr_ass_{seed}"] - df["LB_Braune"]) / df["LB_Braune"]*100
        df[f"GAP_dr_best_{seed}"] = (df[f"GP_dr_best_{seed}"] - df["LB_Braune"]) / df["LB_Braune"]*100

    alg = "dr"
    gp_cols = df.filter(regex=r"^GAP_dr_ass_inst_")
    print("gp_cols", gp_cols.columns)
    df[f"GAP_{alg}_ass_inst_avg"] = gp_cols.mean(axis=1)
    df[f"GAP_{alg}_ass_inst_std"] = gp_cols.std(axis=1)

    gp_cols = df.filter(regex=r"^GAP_dr_ass_")
    print("gp_cols", gp_cols.columns)
    df[f"GAP_{alg}_ass_avg"] = gp_cols.mean(axis=1)
    df[f"GAP_{alg}_ass_std"] = gp_cols.std(axis=1)

    gp_cols = df.filter(regex=r"^GAP_dr_best_")
    print("gp_cols", gp_cols.columns)
    df[f"GAP_{alg}_best_avg"] = gp_cols.mean(axis=1)
    df[f"GAP_{alg}_best_std"] = gp_cols.std(axis=1)

    # alg="pair"
    # gp_cols = df.filter(regex=r"^GP_dr_ass_inst(?!.*runtime)")
    # gp_cols = df.filter(regex=r"^GP_pair_ass_inst(?!.*runtime)")
    # df[f"GP_{alg}_ass_inst_min"] = gp_cols.min(axis=1)
    # df[f"GP_{alg}_ass_inst_avg"] = gp_cols.mean(axis=1)
    # df[f"GP_{alg}_ass_inst_std"] = gp_cols.std(axis=1)
    #
    # gp_cols = df.filter(regex=r"^GP_dr_ass_(?!.*runtime|inst)")
    # gp_cols = df.filter(regex=r"^GP_pair_ass_(?!.*runtime|inst)")
    # df[f"GP_{alg}_ass_min"] = gp_cols.min(axis=1)
    # df[f"GP_{alg}_ass_avg"] = gp_cols.mean(axis=1)
    # df[f"GP_{alg}_ass_std"] = gp_cols.std(axis=1)
    #
    # gp_cols = df.filter(regex=r"^GP_dr_best(?!.*runtime)")
    # gp_cols = df.filter(regex=r"^GP_pair_best(?!.*runtime)")
    # df[f"GP_{alg}_best_min"] = gp_cols.min(axis=1)
    # df[f"GP_{alg}_best_avg"] = gp_cols.mean(axis=1)
    # df[f"GP_{alg}_best_std"] = gp_cols.std(axis=1)

    df.to_excel(f"fjssp-la-details-dr.xlsx")

for_fjssp_la()


#count_gp_rules([0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000])




def  parse_terminals(terminals, rules, count_all=False):
    for rule in rules:
        for key in terminals:
            aparitii = rule.count(key)
            if aparitii>0:
                if count_all:
                    terminals[key] += aparitii
                else:
                    terminals[key] += 1
    print(len(rules),terminals)

import matplotlib.pyplot as plt

def grafic_feature_importance(data, out_file_path):
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel("Features")
    plt.ylabel("Occurrence no.")
    #plt.title("Feature importance / counts")
    plt.xticks(rotation=45, ha="right")
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


def parse_rules_selected(in_files, seeds):
    rules_prefix=["GP_dr_ass", "GP_dr_ass_inst", "GP_dr_best"]
    rules_selection_frequency={}
    for rule_prefix in rules_prefix:
        for seed in seeds:
            rules_selection_frequency[f'{rule_prefix}_{seed}'] = 0

    print("rules_selection_frequency", rules_selection_frequency)


#parse_rules_selected("", [0,200,400,600,800,1000])

def load_gp(df, file_path, load_gp_pair=False):
    with open(file_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)

    if load_gp_pair:
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
        #print(gp_min_df)
        _df = gp_min_df

        #print("_df", df)

    else:
        boms = json_obj["test-data-file-names"]
        makespan = json_obj["agent"]["makespan_tests"]
        apply_gp_time = json_obj["agent"]["running_time"]
        _df  = pd.DataFrame({"Bom": boms, "Makespan": makespan, "RunTime" : apply_gp_time})

    if df.empty:
        df = _df
    else:
        df = pd.concat([df, _df],  ignore_index=True)

    print(df.columns)
    return df

def sign_test(dataset_name, model, path_drsa, path_gp, seeds):
    # dispach rules
    df = pd.read_csv(f"/{path_drsa}/{dataset_name}-dr.csv")

    algo_cols = ['InstanceUpperBound', 'fcfs-b', 'fcfs-max', 'fcfs-min',
                 'fop-b', 'fop-max', 'fop-min',
                 'fopno-b-min', 'fopno-max-min', 'fopno-min-min',
                 'letsa-avg', 'letsa-max', 'letsa-min',
                 'mop-b', 'mop-max', 'mop-min',
                 'mopno-b-min', 'mopno-max-min', 'mopno-min-min']


    df_dr = df.pivot(
        index="Bom",
        columns="Algorithm",
        values="Makespan"
    )
    algo_df = df_dr[algo_cols].replace(-1, np.inf)
    df_dr["dr_min"] = algo_df.min(axis=1)
    df_dr["dr_best_algorithms"] = algo_df.apply(
        lambda row: ", ".join(row.index[row == row.min()]),
        axis=1
    )

    df_sa = pd.read_csv(f"/{path_drsa}/{dataset_name}-sa-ei.csv")
    df_sa["Algorithm"] = df_sa["Algorithm"].replace('SAL-Ei-LM', 'sa')

    df_ass_inst = pd.DataFrame()
    for seed in seeds:
        print(f'{path_gp}/{model}/result_gp_dr_assemble_inst_seed_{seed}.json')
        df_ass_inst = load_gp(df_ass_inst,
                         f'{path_gp}/{model}/result_gp_dr_assemble_inst_seed_{seed}.json', load_gp_pair=True)
    df_ass_inst["Algorithm"] = "gp_dr_ass_inst"

    df_ass = pd.DataFrame()
    for seed in seeds:
        df_ass = load_gp(df_ass,
                               f'{path_gp}/{model}/result_gp_dr_assemble_seed_{seed}.json')
    df_ass["Algorithm"] = "gp_dr_ass"

    df_best = pd.DataFrame()
    for seed in seeds:
        df_best = load_gp(df_best,
                               f'{path_gp}/{model}/result_gp_dr_best_seed_{seed}.json')
        print("df_best", df_best.shape)
    df_best["Algorithm"] = "gp_dr_best"
    print(df_best.columns)

    dfs = {
        "gp_dr_best": df_best,
        "gp_dr_ass": df_ass,
        "gp_dr_ass_inst": df_ass_inst,
        "sa": df_sa,
    }

    common =  pd.concat([df_ass_inst, df_ass, df_best, df_sa, ], ignore_index=True)

    #common = common.dropna()

    n=len(dfs)
    mat = {"gp_dr_ass gp_dr_best":{'ties':0,'win':0,'lose':0},
           "gp_dr_ass sa": {'ties': 0, 'win': 0, 'lose': 0},
           "gp_dr_ass gp_dr_ass_inst": {'ties': 0, 'win': 0, 'lose': 0},

           "gp_dr_best gp_dr_ass":{'ties':0,'win':0,'lose':0},
           "gp_dr_best sa": {'ties': 0, 'win': 0, 'lose': 0},
           "gp_dr_best gp_dr_ass_inst": {'ties': 0, 'win': 0, 'lose': 0},


           "sa gp_dr_ass": {'ties': 0, 'win': 0, 'lose': 0},
           "sa gp_dr_best": {'ties': 0, 'win': 0, 'lose': 0},
           "sa gp_dr_ass_inst": {'ties': 0, 'win': 0, 'lose': 0},

           "gp_dr_ass_inst gp_dr_ass": {'ties': 0, 'win': 0, 'lose': 0},
           "gp_dr_ass_inst gp_dr_best": {'ties': 0, 'win': 0, 'lose': 0},
           "gp_dr_ass_inst sa": {'ties': 0, 'win': 0, 'lose': 0},
           }
    gab_min = {"gp_dr_ass_inst":0,"gp_dr_ass":0, "gp_dr_best":0, "sa":0, }
    gab_avg = {"gp_dr_ass_inst": 0, "gp_dr_ass": 0, "gp_dr_best": 0, "sa": 0, }
    inst_no = 0

    gap_min_1000 = {"gp_dr_ass_inst": 0, "gp_dr_ass": 0, "gp_dr_best": 0, "sa": 0, }
    gap_avg_1000 = {"gp_dr_ass_inst": 0, "gp_dr_ass": 0, "gp_dr_best": 0, "sa": 0, }
    inst_no_1000 =0

    gap_min_500 = {"gp_dr_ass_inst": 0, "gp_dr_ass": 0, "gp_dr_best": 0, "sa": 0, }
    gap_avg_500 = {"gp_dr_ass_inst": 0, "gp_dr_ass": 0, "gp_dr_best": 0, "sa": 0, }
    inst_no_500 = 0

    algs=["gp_dr_ass_inst", "gp_dr_ass", "gp_dr_best","sa", ]
    nr_perechi=6

    print(common.shape)
    constanta = 0.05
    constanta = 0.05 / nr_perechi  # bonfe ...

    print(df_dr.columns)

    for instance in common['Bom'].unique():
        if instance in train_set_instances:
            continue

        inst_no += 1
        df_instance_filtered = common [common['Bom'] == instance]
        insts_details = {}
        #print('df_instance_filtered',df_instance_filtered.shape)
        upperBound = df_dr.loc[instance, "dr_min"]
        runtime_inst = 0
        instance_node_no = df_instance_filtered['Nodes'].unique()[1]
        if instance_node_no >= 500:
            inst_no_500 += 1
        if instance_node_no >= 1000:
            inst_no_1000 += 1

        for alg in algs:
            inst_details = {}
            inst_details['runs'] = df_instance_filtered[df_instance_filtered["Algorithm"]==alg]["Makespan"].head(10)
            inst_details['min'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).min()
            inst_details['mean'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).mean()
            inst_details['std'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["Makespan"].head(10).std()
            inst_details['runtime'] = df_instance_filtered[df_instance_filtered["Algorithm"] == alg]["RunTime"].head(10).sum()
            insts_details[alg]=inst_details
            print(upperBound, inst_details['min'], (upperBound - inst_details['min']) / upperBound)
            gab_min[alg] += (upperBound - inst_details['min']) / upperBound
            gab_avg[alg] += (upperBound - inst_details['mean']) / upperBound
            runtime_inst += inst_details['runtime']
            if instance_node_no >= 500:
                gap_min_500[alg] += (upperBound - inst_details['min']) / upperBound
                gap_avg_500[alg] += (upperBound - inst_details['mean']) / upperBound

            if instance_node_no >= 1000:
                gap_min_1000[alg] += (upperBound - inst_details['min']) / upperBound
                gap_avg_1000[alg] += (upperBound - inst_details['mean']) / upperBound



        #print("runtime_inst",instance, runtime_inst)

        for a, b in itertools.combinations(algs, 2):

            a_details = insts_details[a]
            b_details = insts_details[b]
            # print(a,b)
            # print("A runs:", len(a_details['runs']), "B runs:", len(b_details['runs']))

            d = np.around(np.array(a_details['runs']) - np.array(b_details['runs']), decimals=3)

           # print("algs:",a,b)

            if not np.any(d):
                mat[a+" "+b]['ties'] += 1
                mat[b+" "+a]['ties'] += 1
            else:
                r = wilcoxon(d)
                #print(instance, r,d)
                if r.pvalue < constanta:
                    if np.array(a_details['runs']).mean() < np.array(b_details['runs']).mean():
                        mat[a+" "+b]['win'] += 1
                        mat[b+" "+a]['lose'] += 1
                    else:
                        mat[a+" "+b]['lose'] += 1
                        mat[b+" "+a]['win'] += 1
                else:
                    mat[a + " "+b]['ties'] += 1
                    mat[b + " "+a]['ties'] += 1

    wilcoxon_list = ["gp_dr_ass_inst gp_dr_ass","gp_dr_ass_inst gp_dr_best",
                "gp_dr_ass gp_dr_best","gp_dr_ass_inst sa","gp_dr_ass sa", "gp_dr_best sa"]
    for k in wilcoxon_list:
        e  = mat[k]
        #print(k,";",e['win'],";",e['lose'],";",e['ties'])
        print(k,";",f"({e['win']},{e['ties']},{e['lose']})")

    print("len(common['Bom'].unique())", len(common['Bom'].unique()))
    for alg in algs:
        print(alg, f"gap_min;  {100*gab_min[alg]/inst_no:.2f}")
        print(alg, f"gap_avg;  {100*gab_avg[alg] / inst_no:.2f}")

    print("inst_no_500", inst_no_500)
    if inst_no_500 > 0:
        for alg in algs:
            print(alg, f"gap_min 500;  {100*gap_min_500[alg]/inst_no_500:.2f}")
            print(alg, f"gap_avg 500;  {100*gap_avg_500[alg] /inst_no_500:.2f}")
    else:
        print("No instance with more than 500 nodes")

    print("inst_no_1000", inst_no_1000)
    if inst_no_1000 > 0:
        for alg in algs:
            print(alg, f"gap_min 1000;  {100*gap_min_1000[alg]/inst_no_1000:.2f}")
            print(alg, f"gap_avg 100;  {100*gap_avg_1000[alg] /inst_no_1000:.2f}")
    else:
        print("No instance with more than 500 nodes")

    # for k,e in insts_details.items():
    #     print(k,e)



# dataset=datasets['vdeep-test']
# sign_test(dataset[0],"models-asptrain-large",
#              f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[1]}",
#              f"/Users/flaviamicota/work/scamp-ml/schlaby-asp-gnn-3aprilie/results/{dataset[2]}",
#              [0,200,400,600,800,1000,1500,2000,2500,3000])


