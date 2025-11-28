import concurrent.futures
import random
import operator
import pickle
import numpy as np
from src.utils.logger import Logger
from deap import base, creator, tools, gp, algorithms
from typing import List, Tuple
from src.utils.file_handler.model_handler import ModelHandler


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


def generate_random_value_for_erc():
    return round(random.uniform(-5, 5), 2)

class GP:
    """GP Agent class"""

    def __init__(self, env, config: dict, logger: Logger = None, metadata=None):
        """
        """
        self.logger = logger if logger else Logger(config=config)
        self.env = env

        # set random seed
        if self.env.seed is not None:
            random.seed(self.env.seed)


        self.pop_size: int = config.get('population_size', 10)
        self.halloffame_size: int = config.get('halloffame_size', 1)
        self.cxpb: float = config.get('crossover_probability', 0.5)
        self.mutpb: float = config.get('mutation_probability', 0.3)
        self.ngen: int = config.get('generations_number', 10)
        self.max_expression_depth = config.get('gp_tree_max_depth', 7)
        self.np = config.get('no_parallel_processes', 1)
        self.env_config = config #for saving the solution in file



    def aos_mutation_con(self, toolbox):
        self.OP_NAMES = ["mut_shrink", "mut_uniform", "mut_node_replacement"]
        self.N_OPS = len(self.OP_NAMES)

        self.op_probs = np.ones(self.N_OPS) / self.N_OPS  # probabilități inițiale uniforme
        self.op_rewards = np.zeros(self.N_OPS)  # reward acumulat
        self.op_counts = np.zeros(self.N_OPS) + 1e-9  # câte ori a fost folosit fiecare operator (evităm /0)

        self.ALPHA_PM = 0.8  # “learning rate” pentru Probability Matching

        # înregistrăm operatorii de mutație individual
        toolbox.register("mut_shrink", gp.mutShrink)
        toolbox.register("mut_uniform", gp.mutUniform, expr=toolbox.expr_mut, pset=toolbox.pset)
        toolbox.register("mut_node_replacement", gp.mutNodeReplacement, pset=toolbox.pset)

        self.OP_FUNCS = [
            toolbox.mut_shrink,
            toolbox.mut_uniform,
            toolbox.mut_node_replacement,
        ]

    def adaptive_mutation(self, individual):
        """
        Alege un operator de mutație în mod adaptiv, pe baza op_probs.
        Marchează individul cu op_id, ca să se calculeze reward-ul după evaluare.
        """
        #global op_probs, op_counts

        # alegem operatorul i în funcție de probabilitățile curente
        i = np.random.choice(np.arange(self.N_OPS), p=self.op_probs)
        mut_op = self.OP_FUNCS[i]

        # aplicăm mutația (DEAP mută in-place și întoarce (indiv, ))
        mutant, = mut_op(individual)

        # marcăm ce operator a fost folosit
        mutant.op_id = i

        # numărăm utilizarea
        self.op_counts[i] += 1

        return mutant,

    def config_gp(self, pset):

        #if not hasattr(creator, "FitnessMin"):
        #Creaza functie de optimizare uniobiectiv (weights negativ = minimizare)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        #if not hasattr(creator, "Individual"):
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

        # crossover type - https://deap.readthedocs.io/en/master/api/tools.html
        #gp.cxOnePoint() Selects a random node in parent 1; Selects a random node in parent 2;Swaps the two subtrees
        #gp.cxOnePointLeafBiased(termpb=0.1) Like cxOnePoint, but with a probability of choosing leaf nodes (terminals) rather than function nodes.
        #            Useful when: You want smaller, more stable changes; You want to reduce wild tree growth (bloat); Your trees contain many terminals
        #gp.cxSemantic()
        #toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        # mutation type
        ##gp.mutUniform() Replaces a random subtree with a newly generated one.
        ##gp.mutNodeReplacement() Replaces one random node (function or terminal) with another of the same arity.
        ##gp.mutInsert() Inserts a new subtree above a randomly chosen node.
        ##gp.mutShrink() Replaces a subtree with one of its terminals (or smaller subtree).
        ##gp.mutEphemeral() Randomizes the value of an ephemeral constant.
        #gp.staticLimit() (wrapper) Not a mutation, but wraps a mutation to enforce depth/size limits.
        #mutSemantic
        if not hasattr(toolbox, 'expr') or not hasattr(toolbox, 'pset'):
            raise AttributeError("Toolbox not fully configured for mutation. Missing 'expr' or 'pset'.")
        # toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=toolbox.pset)
        # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))
        self.aos_mutation_con(toolbox)
        toolbox.register("mutate", self.adaptive_mutation)

        # GP parameters
        self.pop = toolbox.population(n=self.pop_size)
        self.hof = tools.HallOfFame(self.halloffame_size)


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

        mstats = tools.MultiStatistics(fitness=stats_fit,
                                            #size=stats_size,
                                            xbest_ind=stats_best_ind_obj
                                            )#tools.MultiStatistics(fitness=stats_fit)

        return toolbox, mstats



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
                makespan = float('inf')

            if makespan != float('inf'):
                total_combined_score += makespan
                num_valid_instances_evaluated += 1
        if num_valid_instances_evaluated == 0:
            print("Infinit!!!!!!!!!!!!")
            return (float('inf'),)
        #print(" makespan mediu individ", total_combined_score / num_valid_instances_evaluated)
        return (total_combined_score / num_valid_instances_evaluated,)


def update_operator_probs(self):
    global op_probs, op_rewards, op_counts

    avg_rewards = op_rewards / op_counts
    if np.all(avg_rewards == 0):
        op_probs = np.ones(self.N_OPS) / self.N_OPS
    else:
        target_probs = avg_rewards / np.sum(avg_rewards)
        uniform = np.ones(self.N_OPS) / self.N_OPS
        op_probs = (1 - self.ALPHA_PM) * uniform + self.ALPHA_PM * target_probs

    op_rewards[:] = 0.0
    op_counts[:] = 1e-9

def eaSimpleAOS(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # evaluate initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # main loop
    for gen in range(1, ngen + 1):
        # select + vary
        offspring = tools.selTournament(population, len(population), tournsize=3)
        offspring = list(map(toolbox.clone, offspring))

        # salvăm fitness-ul părintelui pentru reward
        for ind in offspring:
            ind.parent_fitness = ind.fitness.values[0]

        # crossover + mutation
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        for ind in offspring:
            if random.random() < mutpb:
                ind, = toolbox.mutate(ind)
                del ind.fitness.values

        # evaluate newly created individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # ====== AOS: calculăm reward per operator ======
        for ind in offspring:
            if hasattr(ind, "op_id"):
                parent_fit = ind.parent_fitness
                child_fit = ind.fitness.values[0]
                # ex: pentru minimizare
                delta = parent_fit - child_fit
                reward = max(0.0, delta)
                self.op_rewards[ind.op_id] += reward

        # actualizăm probabilitățile operatorilor
        self.update_operator_probs()
        # ===============================================

        # hall of fame + log
        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = toolbox.select(offspring, len(population))

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test = None) -> None:
        """
        Learn over n environment instances or n timesteps. Break depending on which condition is met first
        One learning iteration consists of collecting rollouts and training the networks

        :param total_instances: Instance limit
        :param total_timesteps: - not used, kept for compatibility with the framework
        :param intermediate_test: - not used, kept for compatibility with the framework

        """
        toolbox, mstats = self.config_gp(GP.config_gp_expression())

        # Create the logbook
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + mstats.fields  # IMPORTANT!


        print("\n--- Start GP ---")
        final_pop, logbook = self.eaSimpleAOS(#algorithms.eaSimple(
            self.pop, toolbox,
            cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=self.ngen,
            stats=mstats,
            halloffame=self.hof,
            verbose=True
        )


        #logbook = mstats.compile(final_pop)
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
        print("Best individual:",self.best_ind)
        self.save(ModelHandler.get_best_model_path(self.env_config))

    @classmethod
    def config_gp_expression(cls) -> base.Toolbox:
        """
        Describe the GP individual
        :return: configured DEAP object
        """
        # PrimitiveSet "MAIN"
        pset = gp.PrimitiveSet("MAIN", 9)

        # Score function arguments:
        pset.renameArguments(ARG0='O_MeanPT')    # OP: Mean processing time: Estimates operation duration.
        pset.renameArguments(ARG1='O_MinPT')     # OP: Minimum processing time: Highlights the quickest possible execution time
        pset.renameArguments(ARG2='O_Flex')      # OP:  Ratio of machines that are eligible for Oij to total machine number
        pset.renameArguments(ARG3='E_PT')        #Edge (op, machine): Processing time p_{ik}  of operation i on machine k
        pset.renameArguments(ARG4='E_PT_maxPT')  #Edge (op, machine): Ratio of p_{ik} to the maximum processing time of p_{il}  l=1,M_i  (M_i= total number of machines on which op i can be executed)
        pset.renameArguments(ARG5='E_PT_maxMPT') #Edge (op, machine): Ratio of p_{ik} to the maximum processing time of p_{lk}  l=1,N _k (N_k= total number of operations that can be executed on machine k)
        pset.renameArguments(ARG6='M_RT')        # Machine: Last operation completion time t_{last}: Determines machine availability.
        pset.renameArguments(ARG7='M_OP')        # Machine: Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
        pset.renameArguments(ARG8='M_UT')        # Machine: Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.

        # Non-terminals
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addPrimitive(operator.sub, 2, name="sub")
        pset.addPrimitive(operator.mul, 2, name="mul")
        pset.addPrimitive(protected_div, 2, name="protected_div")
        pset.addPrimitive(operator.neg, 1, name="neg")
        pset.addPrimitive(min, 2, name="min")
        pset.addPrimitive(max, 2, name="max")

        # Terminals
        pset.addTerminal(1.0, name="oneF")
        pset.addEphemeralConstant("ERC", generate_random_value_for_erc)
        #pset.addEphemeralConstant("ERC", lambda: random.random(), float) - forma generala

        return pset


    def save(self, file: str) -> None:
        """
        Save model as pickle file

        :param file: Path under which the file will be saved

        :return: None

        """
        params_dict = self.__dict__.copy()
        del params_dict['logger']
        data = {
            "best_ind": self.best_ind
        }

        print("save in file:", data)
        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)

    @classmethod
    def load(cls, file: str, config: dict = None, logger: Logger = None):
        """
        Loads a GP-expression object according to the parameters saved in file.pkl

        :param file: Path and filename (without .pkl) of your saved model pickle file
        :param config: kept for compatibility with other agents
        :param logger: Logger

        :return: toolbox.individual or toolbox.expr VERIFY!!!

        """
        print("GP-load()", file)
        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)

            print("load() data: ", data)

        toolbox = base.Toolbox()
        toolbox.register("compile", gp.compile, pset=GP.config_gp_expression())

        return toolbox.compile(expr=data['best_ind'])


    def predict(self, state = None, observation = None, deterministic: bool = True):
        #este functia select_action
        print("predict --- state", state)
        return #FM-intoarce  log_prob in loc de state
