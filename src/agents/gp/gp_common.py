import random
import concurrent.futures
import pickle
from operator import truediv

from deap import base, creator, tools, gp, algorithms
import operator
from typing import List, Tuple

from src.utils.file_handler.model_handler import ModelHandler
from src.utils.logger import Logger
from src.agents.gp.simpleTree import tree_str, infix_str, simplify_individual


class GPBase:
    """GP Agent class"""

    def __init__(self, env, config: dict, logger: Logger = None, metadata=None):
        """
        """
        self.logger = logger if logger else Logger(config=config)
        self.env = env

        # set random seed
        if self.env.seed is not None:
            random.seed(self.env.seed)

        if config.get('aos_type', 'aos'):
            self.use_aos = True
            self.use_qlearning = False
        else:
            self.use_aos = False
            self.use_qlearning = True

        self.pop_size: int = config.get('population_size', 10)
        self.halloffame_size: int = config.get('halloffame_size', 1)
        self.cxpb: float = config.get('crossover_probability', 0.5)
        self.mutpb: float = config.get('mutation_probability', 0.3)
        self.ngen: int = config.get('generations_number', 10)
        self.max_expression_depth = config.get('gp_tree_max_depth', 7)
        self.np = config.get('no_parallel_processes', 1)
        self.env_config = config #for saving the solution in file

    def multi_instance_fitness(self, individual: gp.PrimitiveTree,
                           toolbox: base.Toolbox,
                           ) -> Tuple[float,]:
        # if len(self.hof):
        #     print('best so far',self.hof[0], len(self.hof))
        #print("multi_instance_fitness() - start_eval", individual )
        if individual is None:
            return (float('inf'),)

        try:
            priority_func = toolbox.compile(expr=individual)
        except Exception as e:
            return (float('inf'),)

        #print(individual)
        total_combined_score = 0.0
        num_valid_instances_evaluated = 0
        self.env.current_instance_index = -1 #
        #print("multi_instance_fitness() call")
        for inst_no in range(self.env.instances_no):
            try:
                makespan = self.env.evaluate_instance(priority_func)
                #print(makespan, end="+")
            except Exception as e_eval:
                print("EROARE EVAL: ", e_eval)
                import traceback
                traceback.print_exc()

                makespan = float('inf')

            if makespan != float('inf'):
                total_combined_score += makespan
                num_valid_instances_evaluated += 1
        if num_valid_instances_evaluated == 0:
            print("Infinit!!!!!!!!!!!!")
            return (float('inf'),)
        #print(" makespan mediu individ", total_combined_score / num_valid_instances_evaluated)
        return (total_combined_score / num_valid_instances_evaluated,)

    @classmethod
    def config_gp_expression(cls) -> base.Toolbox:
        """
        Describe the GP individual
        :return: configured DEAP object
        """
        # PrimitiveSet "MAIN"
        pset = gp.PrimitiveSetTyped("MAIN", [float,float,float,float, float,float,float,float,float,float,float,float], float)#9

        # Score function arguments:
        pset.renameArguments(ARG0='O_MeanPT')  # OP: Mean processing time: Estimates operation duration.
        pset.renameArguments(
            ARG1='O_MinPT')  # OP: Minimum processing time: Highlights the quickest possible execution time
        pset.renameArguments(ARG2='O_Flex')  # OP:  Ratio of machines that are eligible for Oij to total machine number
        pset.renameArguments(ARG3='E_PT')  # Edge (op, machine): Processing time p_{ik}  of operation i on machine k
        pset.renameArguments(
            ARG4='E_PT_maxPT')  # Edge (op, machine): Ratio of p_{ik} to the maximum processing time of p_{il}  l=1,M_i  (M_i= total number of machines on which op i can be executed)
        pset.renameArguments(
            ARG5='E_PT_maxMPT')  # Edge (op, machine): Ratio of p_{ik} to the maximum processing time of p_{lk}  l=1,N _k (N_k= total number of operations that can be executed on machine k)
        pset.renameArguments(
            ARG6='M_RT')  # Machine: Last operation completion time t_{last}: Determines machine availability.
        pset.renameArguments(
            ARG7='M_OP')  # Machine: Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
        pset.renameArguments(
            ARG8='M_UT')  # Machine: Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
        pset.renameArguments(ARG9="O_Path_OpNo")
        pset.renameArguments(ARG10="O_Path_MinLen")
        pset.renameArguments(ARG10="O_WT") #current makespan - operation.release_time

        # Non-terminals
        pset.addPrimitive(operator.add, [float,float], float)
        pset.addPrimitive(operator.sub, [float,float], float)
        pset.addPrimitive(operator.mul, [float,float], float)
        pset.addPrimitive(protected_div, [float,float], float)
        pset.addPrimitive(protected_if, [bool,float, float], float)
        pset.addPrimitive(operator.neg, [float] , float)
        pset.addPrimitive(min, [float,float], float)
        pset.addPrimitive(max, [float,float], float)

        #pset.addPrimitive(gt, [float, float], bool)
        pset.addPrimitive(lt, [float, float], bool)


        # Terminals
        pset.addTerminal(True, bool)
        pset.addTerminal(False, bool)
        pset.addTerminal(1.0, float)
        pset.addEphemeralConstant("ERC", generate_random_value_for_erc, float)
        # pset.addEphemeralConstant("ERC", lambda: random.random(), float) - forma generala

        return pset


    def config_statistics(self):
        # redefinire functii din statistici deoarece pot contine valori inf - de vazut daca si acum mai e cazul
        safe_avg = lambda x: sum(xi for xi in x if xi != float('inf')) / len(
            [xi for xi in x if xi != float('inf')]) if len(
            [xi for xi in x if xi != float('inf')]) > 0 else 0.0
        safe_min = lambda x: min(xi for xi in x if xi != float('inf')) if any(
            xi != float('inf') for xi in x) else float(
            'inf')
        safe_max = lambda x: max(xi for xi in x if xi != float('inf')) if any(
            xi != float('inf') for xi in x) else float(
            '-inf')

        def safe_std(x_list):
            finite_vals = [xi for xi in x_list if xi != float('inf')]
            if len(finite_vals) < 2: return 0.0
            mean_val = sum(finite_vals) / len(finite_vals)
            return (sum((xi - mean_val) ** 2 for xi in finite_vals) / len(finite_vals)) ** 0.5

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else float('inf'))
        stats_fit.register("avg", safe_avg)
        stats_fit.register("std", safe_std)
        stats_fit.register("min", safe_min)
        stats_fit.register("max", safe_max)

        stats_size = tools.Statistics(len)
        stats_size.register("avg", lambda x: sum(x) / len(x) if len(x) > 0 else 0.0)
        stats_size.register("std", safe_std)
        stats_size.register("min", safe_min)
        stats_size.register("max", safe_max)

        # `min` va alege individul bazat pe `ind.fitness`
        stats_best_ind_obj = tools.Statistics(key=lambda ind: ind)
        stats_best_ind_obj.register("best", lambda pop_list: min(pop_list, key=lambda
            ind: ind.fitness.values[0] if ind.fitness.valid else float('inf')))

        return tools.MultiStatistics(fitness=stats_fit,
                                       # size=stats_size,
                                       xbest_ind=stats_best_ind_obj
                                       )  # tools.MultiStatistics(fitness=stats_fit)

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Learn over n environment instances or n timesteps. Break depending on which condition is met first
        One learning iteration consists of collecting rollouts and training the networks

        :param total_instances: Instance limit
        :param total_timesteps: - not used, kept for compatibility with the framework
        :param intermediate_test: - not used, kept for compatibility with the framework

        """
        toolbox = self.config_gp(GPBase.config_gp_expression())
        mstats = self.config_statistics()

        # Create the logbook
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + mstats.fields  # IMPORTANT!

        #call GP variant
        final_pop, logbook = self.runGP(toolbox, mstats)

        # logbook = mstats.compile(final_pop)
        print("----logbook-----\n")

        print("\n--- Best Individual per Generation (from Logbook) ---")
        if logbook:
            print(f"Gen\t{'MinFitness':<15}\tBest Individual Tree of Generation")
            print("-" * 80)
            for gen_data in logbook:
                gen_num = gen_data["gen"]

                min_fitness_val = gen_data.get("fitness", {}).get("min", float('inf'))
                best_ind_tree_of_gen = None
                if "best_individual" in gen_data and "best" in gen_data["best_individual"]:
                    best_ind_tree_of_gen = gen_data["best_individual"]["best"]
                print(
                    f"{gen_num}\t{min_fitness_val:<15.4f}\t{str(best_ind_tree_of_gen) if best_ind_tree_of_gen else 'N/A'}")
        else:
            print("Logbook is empty or not generated.")

        print("\nGenetic program finished.")

        self.best_ind = self.hof[0]
        for el in self.hof:
            print("hof", el)
        print("Best individual:", tree_str(self.best_ind))
        print(infix_str(self.best_ind))  # vezi formula matematică

        simp = simplify_individual(self.best_ind, toolbox.pset)
        print("Simplified:")
        print(infix_str(simp))

        self.save(ModelHandler.get_best_model_path(self.env_config))


    def runGP(self, toolbox, mstats):
        #implemented in subclassesy
        pass

    def config_gp(self, pset):

        if not hasattr(creator, "FitnessMin"):
            #Creaza functie de optimizare uniobiectiv (weights negativ = minimizare)
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5) #(1,5) adaincimea in populatia initiala
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        if self.np > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.np if self.np > 0 else None)
            toolbox.register("map", executor.map)

        toolbox.pset = pset

        # register fitness function
        toolbox.register("evaluate",
                         self.multi_instance_fitness, #the fitness function
                         toolbox=toolbox)

        # selection type
        toolbox.register("select", tools.selTournament, tournsize=3)


        self.config_gp_variation_operators(toolbox)

        # GP parameters
        self.pop = toolbox.population(n=self.pop_size)
        self.hof = tools.HallOfFame(self.halloffame_size)

        return toolbox

    def config_gp_variation_operators(self, toolbox):
        pass


    @classmethod
    def load(cls, file: str, config: dict = None, logger: Logger = None):
        """
        Loads a GP-expression object according to the parameters saved in file.pkl

        :param file: Path and filename (without .pkl) of your saved model pickle file
        :param config: kept for compatibility with other agents
        :param logger: Logger

        :return: the compiled tree

        """
        print("GP-load()", file)
        evaluation_type = config.get('evaluation_type', 'best')
        print("load-self.evaluation_type", evaluation_type)

        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)
            print("load() data: ", data)

        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=GPBase.config_gp_expression())

        if evaluation_type == 'best':
            best = data['best_ind']
            return toolbox.compile(expr=best), best
        elif evaluation_type == 'assemble' or evaluation_type == 'assemble-test':
            aassemble_fct = []

            for el in data['hof']:
                aassemble_fct.append(toolbox.compile(expr=el))
                print(el,aassemble_fct)
            return aassemble_fct, data['hof']


    def predict(self, state=None, observation=None, deterministic: bool = True):
        return

    def save(self, file: str) -> None:
        """
        Save model as pickle file

        :param file: Path under which the file will be saved

        :return: None

        """
        params_dict = self.__dict__.copy()
        del params_dict['logger']
        data = {
            "best_ind": self.best_ind,
            "hof": self.hof
        }

        print("save in file:", str(data["best_ind"]))
        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)



def generate_random_value_for_erc():
    return round(random.uniform(-5, 5), 2)


def protected_if(cond, true_expr, false_expr):
    """
    Evaluare IF .
    - cond > 0   → returnează true_expr
    - cond <= 0  → returnează false_expr
    """
    try:
        if cond is None:
            return false_expr

        # tratăm NaN ca False
        if isinstance(cond, float) and (cond != cond):
            return false_expr

        return true_expr if cond > 0 else false_expr
    except Exception:
        return false_expr

def gt(x, y):
    return x > y

def lt(x, y):
    return x < y

def protected_div(a: float, b: float) -> float:
    """
    Diviziune protejată pentru a evita erorile de împărțire la zero.
    Returnează `a` dacă `b` este foarte aproape de zero, altfel `a / b`.
    """
    # Daca b e 0, am putea returna o valoare mare daca a e pozitiv,
    # sau a insusi, sau 1.0. Alegerea depinde de cum vrem sa penalizam/interpretam.
    # Varianta initiala `else a` poate fi problematica daca `a` e mic si `b` e aproape de 0.
    # O valoare mare ar putea fi mai sigura pentru a evita prioritati neasteptat de mari.
    if abs(b) < 1e-9:
        if a > 1e-9: return 1e9  # Numar mare pozitiv
        if a < -1e-9: return -1e9  # Numar mare negativ
        return 0.0  # Daca si a e 0
    return a / b



