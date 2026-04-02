import random
import concurrent.futures
import pickle
import traceback
import json
import numpy as np

from deap import base, creator, tools, gp
import operator
import copy
from typing import Tuple, AnyStr, Literal, List, Union
from enum import Enum

from src.utils.file_handler.model_handler import ModelHandler
from src.utils.logger import Logger
from src.agents.gp.simpleTree import infix_str
from src.agents.gp.util import protected_div, protected_if, generate_random_value_for_erc, lt
from src.agents.gp.simpleTree import  simplify_individual

class OperatorSpec:
    def __init__(self, name : AnyStr, optype : Literal['mutation', 'crossover'], func):
        """
        name   : string for debug/log
        optype : "mutation" or "crossover"
        func   : DEAP function registered in toolbox
        """

        self.name = name
        self.type = optype
        self.func = func

    def apply(self, ind1, ind2=None):
        """
        Call the operator in a unified way.
        - mutation: receives 1 individual and returns  (new_ind, None)
        - crossover: receives 2 individuals and returns (child1, child2)
        """
        if self.type == 'mutation':
            (new_ind,) = self.func(ind1)
            return new_ind, None

        elif self.type == 'crossover':
            if ind2 is None:
                raise ValueError(f"{self.name}: crossover needs 2 inds")
            child1, child2 = self.func(ind1, ind2)
            # guard
            if callable(child1) or callable(child2):
                raise TypeError(f"{self.name}: returned callables, not inds")
            return child1, child2
        else:
            raise ValueError(f"Unknown operator type: {self.type}")

class GPBase:
    """GP Agent class"""

    class AOSType(str, Enum):
        """
        Adaptive operator selection types
        """
        AOS = "aos"
        AOS_MEMORY = "aos-memory"
        RANDOM = "random"

    class RuleEvaluationType(str, Enum):
        """
        Adaptive operator selection types
        """
        BEST = "best"
        ASSEMBLE = "assemble"
        ASSEMBLE_INSTANCES = "assemble-instances"

    def __init__(self, env, config: dict, logger: Logger = None, metadata=None):
        """
        """
        self.logger = logger if logger else Logger(config=config)
        self.env = env

        # set random seed
        if self.env.seed is not None: random.seed(self.env.seed)

        self.aos_type : GPBase.AOSType =  GPBase.AOSType(config.get('gp_aos_type', 'random'))

        self.pop_size: int = config.get('gp_population_size', 10)
        self.halloffame_size: int = config.get('gp_halloffame_size', 1)
        self.variation_probability: float = config.get('gp_population_variation', 0.95)
        self.ngen: int = config.get('gp_generations_number', 10)
        self.max_expression_depth: int = config.get('gp_tree_max_depth', 7)
        self.gp_tree_initial_max_depth: int = config.get('gp_tree_initial_max_depth', 3)
        self.simplify_frequency: int = config.get('gp_simplify_frequency', 10)
        self.tournament_size: int = config.get('gp_tournament_size', 3)
        self.np: int = config.get('no_parallel_processes', 1)
        self.env_config = config #for saving the solution in file

    def multi_instance_fitness(self, individual: Union[gp.PrimitiveTree,
                        List [gp.PrimitiveTree]],
                        toolbox: base.Toolbox,
                        ) -> Tuple[float,]:
        """
        Evaluate an individual
        :param individual: an individual from the population (depending on the GP type 1Tree or 2Tress)
        :param toolbox: DEAP toolbox
        :return: the fitness value (mean makespan on all train instances) and the list with all saved decision states
        """
        if individual is None:
            return (float('inf'),)
        try:
            if isinstance(individual, gp.PrimitiveTree):
                #individual contains 1 tree
                priority_func = [toolbox.compile(expr=individual)]
            else:
                # individual contains 2 trees
                priority_func = [toolbox.compile_disp(expr=individual[0]), toolbox.compile_route(expr=individual[1])]
        except Exception as e:
            traceback.print_exc()
            return (float('inf'),)

        total_combined_score = 0.0
        num_valid_instances_evaluated = 0
        self.env.current_instance_index = -1

        #all_decision_states = []
        for inst_no in range(self.env.instances_no):
            makespan = float('inf')
            try:
                makespan = self.env.evaluate_instance(priority_func, self.individual_trees_no)
                 #, decision_states)

                #all_decision_states.append(decision_states)
            except Exception as e_eval:
                traceback.print_exc()

            if makespan != float('inf'):
                total_combined_score += makespan
                num_valid_instances_evaluated += 1
            else:
                traceback.print_exc()

        if num_valid_instances_evaluated == 0:
            print("Infinity!!!!!!!!!!!!")
            return (float('inf'),)

        return (total_combined_score / num_valid_instances_evaluated,)#, all_decision_states

    def configure_terminals(self):
        """
        Set terminals for GP individual
        :return: pset
        """
        pass

    @classmethod
    def configure_non_terminals_and_common_primitive(cls, pset: gp.PrimitiveSetTyped) -> gp.PrimitiveSetTyped:
        """
       Set non-terminals for GP individual and common primitives (constants, ERC)
       :param pset: - GP primitive set
       :return: pset: - GP primitive set
       """

        # Non-terminals
        pset.addPrimitive(operator.add, [float,float], float)
        pset.addPrimitive(operator.sub, [float,float], float)
        pset.addPrimitive(operator.mul, [float,float], float)
        pset.addPrimitive(protected_div, [float,float], float)
        pset.addPrimitive(protected_if, [bool,float, float], float)
        pset.addPrimitive(operator.neg, [float] , float)
        pset.addPrimitive(min, [float,float], float)
        pset.addPrimitive(max, [float,float], float)
        pset.addPrimitive(lt, [float, float], bool)

        # Terminals
        pset.addTerminal(True, bool)
        pset.addTerminal(False, bool)
        pset.addEphemeralConstant("ERC", generate_random_value_for_erc, float)

        return pset


    def config_statistics(self):
        # redefine statistic functions
        safe_avg = lambda x: sum(xi for xi in x if xi != float('inf')) / len(
            [xi for xi in x if xi != float('inf')]) if len(
            [xi for xi in x if xi != float('inf')]) > 0 else 0.0
        safe_min = lambda x: min(xi for xi in x if xi != float('inf')) if any(
            xi != float('inf') for xi in x) else float('inf')
        safe_max = lambda x: max(xi for xi in x if xi != float('inf')) if any(
            xi != float('inf') for xi in x) else float('-inf')

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

        stats_best_ind_obj = self.register_individual_statistic(tools)
        return tools.MultiStatistics(fitness=stats_fit,
                                     xbest_ind=stats_best_ind_obj)

    def register_individual_statistic(self, tools):
        """
        Register statistic regarding the best generation individual
        :param tools: DEAP tools
        :return: the DEAP statistic
        """
        pass

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Learn over n environment instances or n timesteps. Break depending on which condition is met first
        One learning iteration consists of collecting rollouts and training the networks

        :param total_instances:   - not used, kept for compatibility with the framework
        :param total_timesteps:   - not used, kept for compatibility with the framework
        :param intermediate_test: - not used, kept for compatibility with the framework

        """
        toolbox = self.config_gp()
        mstats = self.config_statistics()

        # Create the logbook
        logbook = tools.Logbook()

        #call GP variant
        final_pop, logbook = self.runGP(toolbox, mstats)


        print("----logbook-----\n")
        print("\n--- Best Individual per Generation (from Logbook) ---")
        if logbook:
            print(f"Gen\t{'MinFitness':<15}\tBest Individual Tree of Generation")
            print("-" * 80)

            fit_ch = logbook.chapters["fitness"]
            best_ch = logbook.chapters["xbest_ind"]

            for i, root in enumerate(logbook):
                gen = root["gen"]
                #nevals = root["nevals"]

                f = fit_ch[i]
                b = best_ch[i]

                if 'operation_selection_rule' in b.keys():
                    print(f"{gen}\t{f['min']:<15.4f}\t{str(b['operation_selection_rule'])}  {str(b['machine_selection_rule'])}")
                else:
                    print(f"{gen}\t{f['min']:<15.4f}\t{str(b['best_tree'])}")
        else:
            print("Logbook is empty or not generated.")

        print("\nGenetic program finished.")

        self.best_ind = self.hof[0]
        self.display_hof(self.hof, toolbox)

        self.save(ModelHandler.get_best_model_path(self.env_config))
        return self.best_ind.fitness.values[0]

    def display_hof(self, hof, toolbox):
        pass

    def runGP(self, toolbox, mstats):
        #implemented in subclasses
        pass


    def operator_selection_strategy_configuration(self):
        """
        Prepare data stuctures to store operator selection information
        :return:
        """
        self.N_OPS = len(self.OP_SPECS)

        if self.aos_type is self.AOSType.AOS: #self.use_aos:
            self.op_probs = np.ones(self.N_OPS) / self.N_OPS  # initial probabilities - uniform distributed
            self.op_rewards = np.zeros(self.N_OPS)  # accumulated reward
            self.op_counts = np.zeros(self.N_OPS) + 1e-9  # how many times each operator was used (avoid /0)
            self.ALPHA_PM = 0.8  # “learning rate” for Probability Matching

        elif self.aos_type is self.AOSType.AOS_MEMORY: #self.use_qlearning:
            self.Q_ops = np.zeros(self.N_OPS)  # Q-value for each operator
            self.ALPHA_Q = 0.2  # learning rate (can be ajusted)
            self.EPSILON_Q = 0.1  # exploration (10% random)

    def update_operator_probs(self):
        """
        update information for AOSType.AOS
        :return:
        """
        avg_rewards = self.op_rewards / self.op_counts
        if np.all(avg_rewards == 0):
            self.op_probs = np.ones(self.N_OPS) / self.N_OPS
        else:
            target_probs = avg_rewards / np.sum(avg_rewards)
            uniform = np.ones(self.N_OPS) / self.N_OPS
            self.op_probs = (1 - self.ALPHA_PM) * uniform + self.ALPHA_PM * target_probs

        # print(f"op_rewards: {self.op_rewards}")
        # print(f"op_counts:  {self.op_counts}")
        # print(f"op_probs:   {self.op_probs}")
        self.op_rewards[:] = 0.0
        self.op_counts[:] = 1e-9

    def config_gp(self):
        # import inspect
        # print(f"GPBase defined in: {inspect.getfile(GPBase)}, self class {inspect.getfile(self.__class__)}" )

        #minimization uni-objective function
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        toolbox = base.Toolbox()

        toolbox = self.config_individual(toolbox)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        if self.np > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.np if self.np > 0 else None)
            toolbox.register("map", executor.map)

        # selection type
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        self.OP_SPECS = self.config_gp_variation_operators(toolbox)

        self.operator_selection_strategy_configuration()

        # GP parameters
        self.pop = toolbox.population(n=self.pop_size)
        self.hof = tools.HallOfFame(self.halloffame_size)

        toolbox.register("clone", copy.deepcopy)

        return toolbox

    def config_gp_variation_operators(self, toolbox):
        # implemented in subclasses
        pass

    def config_individual(self, toolbox):
        # implemented in subclasses
        pass

    def eaSimpleGP(self, population, toolbox, varpb, ngen, stats=None, halloffame=None, verbose=__debug__):

        logbook = tools.Logbook()

        # evaluate initial population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        evaluation_results = toolbox.map(toolbox.evaluate, invalid_ind)

        # for ind, (fit, decision_states) in zip(invalid_ind, evaluation_results):
        #     print(ind, fit, decision_states)
        #     ind.fitness.values = fit
        #     ind.decision_states = decision_states

        for ind, fit in zip(invalid_ind, evaluation_results):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose: print(logbook.stream)

        # main loop
        for gen in range(1, ngen + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for i, ind in enumerate(offspring):
                if callable(ind) or not hasattr(ind, "fitness"):
                    raise TypeError(f"Selected offspring invalid at {i}: {type(ind)} {ind}")

            # save parent fitness for reward
            for ind in offspring:
                if hasattr(ind, "op_id"): delattr(ind, "op_id")
                ind.parent_fitness = ind.fitness.values[0]

            for i in range(0, len(offspring) - 1, 2):
                ind1 = offspring[i]
                ind2 = offspring[i + 1]

                if  random.random() >= varpb:
                    # no operator is applied on this pair
                    continue

                # choose an operator
                if self.aos_type is self.AOSType.AOS: #self.use_aos:
                    op_idx = np.random.choice(np.arange(self.N_OPS), p=self.op_probs)
                elif self.aos_type is self.AOSType.AOS_MEMORY: #self.use_qlearning:
                    #Selecte an operator using a ε-greedy policy over Q_ops.
                    if random.random() < self.EPSILON_Q: # explore: select random an operator
                        op_idx = random.randrange(self.N_OPS)
                    else: #exploit: select the operator with maximal Q
                        op_idx =  int(np.argmax(self.Q_ops))
                else: #no strategy to select operator
                    op_idx = np.random.choice(np.arange(self.N_OPS))

                op_spec = self.OP_SPECS[op_idx]

                if op_spec.type == "crossover":
                    child1, child2 = op_spec.apply(ind1, ind2)
                else:
                    child1, _ = op_spec.apply(ind1)
                    child2, _ = op_spec.apply(ind2)

                #save information for AOS
                child1.op_id = op_idx
                child2.op_id = op_idx

                offspring[i], offspring[i + 1] = child1, child2
                if hasattr(child1, "fitness"):
                    if hasattr(child1.fitness, "values"):
                        del child1.fitness.values
                if hasattr(child2, "fitness"):
                    if hasattr(child2.fitness, "values"):
                        del child2.fitness.values

            #simplify algebraic population
            if gen % self.simplify_frequency == 0:
                offspring = self.simplify_population(offspring, toolbox)

            # evaluate newly created individuals
            for i, ind in enumerate(offspring):
                # delete previous saved decision states
                #ind.decision_states = []
                if not hasattr(ind, "fitness"):
                    raise TypeError(f"Invalid offspring at index {i}: type={type(ind)} value={ind}")

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            evaluation_results = toolbox.map(toolbox.evaluate, invalid_ind)
            # for ind, (fit, decision_states) in zip(invalid_ind, evaluation_results):
            #     ind.fitness.values = fit
            #     ind.decision_states = decision_states

            for ind, fit in zip(invalid_ind, evaluation_results):
                ind.fitness.values = fit

            # simplify behaviour population
            # for indx, ind in enumerate(offspring):
            #     print(f"individual {indx} saved decision states {ind.decision_states}")

            # ====== Update AOS information ======
            for ind in offspring:
                if hasattr(ind, "op_id"):
                    #parent_fit - child_fit
                    reward = max(0.0, ind.parent_fitness - ind.fitness.values[0])

                    o = ind.op_id  # index of used operator
                    if self.aos_type is self.AOSType.AOS:#self.use_aos:
                        self.op_rewards[o] += reward
                        self.op_counts[o] += 1.0

                    elif self.aos_type is self.AOSType.AOS_MEMORY:#self.use_qlearning:
                        # update Q(o) ← Q(o) + α * (r - Q(o))
                        self.Q_ops[o] = self.Q_ops[o] + self.ALPHA_Q * (reward - self.Q_ops[o])

            if self.aos_type is self.AOSType.AOS:
                #update operator probabilities
                self.update_operator_probs()
            # ===============================================

            # hall of fame + log
            if halloffame is not None: halloffame.update(offspring)
            population[:] = toolbox.select(offspring, len(population))

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def simplify_population(self, offspring, toolbox):
        pass

    def runGP(self, toolbox, mstats):
        print("\n--- Start GP ---")
        final_pop, logbook = self.eaSimpleGP(
            self.pop, toolbox,
            varpb=self.variation_probability,
            ngen=self.ngen,
            stats=mstats,
            halloffame=self.hof,
            verbose=True
        )
        return final_pop, logbook

    @classmethod
    def load(cls, file: str, config: dict = None, logger: Logger = None):
        """
        Loads a GP-expression object according to the parameters saved in file.pkl

        :param file: Path and filename (without .pkl) of your saved model pickle file
        :param config: kept for compatibility with other agents
        :param logger: Logger

        :return: the compiled tree

        """
        evaluation_type = GPBase.RuleEvaluationType(config.get('evaluation_type', 'best'))
        print("Evaluation_type", evaluation_type)

        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)

        toolbox = base.Toolbox()
        from src.agents.gp.gp_1tree import GP_One_Tree
        from src.agents.gp.gp_2trees import GP_Two_Trees
        if issubclass(cls, GP_One_Tree):
            toolbox.register("compile", gp.compile, pset=cls.configure_terminals())
        elif issubclass(cls, GP_Two_Trees):
            pset_disp, pset_route = cls.configure_terminals()

            toolbox.register("compile_disp", gp.compile, pset=pset_disp)
            toolbox.register("compile_route", gp.compile, pset=pset_route)

        if evaluation_type is GPBase.RuleEvaluationType.BEST:
            best = data['best_ind']
            # let it here to avoid recursive file inclusion
            from src.agents.gp.gp_1tree import GP_One_Tree
            from src.agents.gp.gp_2trees import GP_Two_Trees
            if issubclass(cls, GP_One_Tree):
                return [toolbox.compile(expr=best)], best
            elif issubclass(cls, GP_Two_Trees):
                return [toolbox.compile_disp(expr=best[0]),
                        toolbox.compile_route(expr=best[1])], best
        elif (evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE or
              evaluation_type is GPBase.RuleEvaluationType.ASSEMBLE_INSTANCES):
            assemble_fct = []

            pset_dispatch, pset_machines = GP_Two_Trees.configure_terminals()
            for el in data['hof']:
                #let it here to avoid recursive file inclusion
                from src.agents.gp.gp_1tree import GP_One_Tree
                from src.agents.gp.gp_2trees import GP_Two_Trees

                if issubclass(cls, GP_One_Tree):
                    assemble_fct.append([toolbox.compile(expr=el)])
                elif issubclass(cls, GP_Two_Trees):
                    el[0] = simplify_individual(el[0], pset_dispatch)
                    el[1] = simplify_individual(el[1], pset_machines)

                    assemble_fct.append([toolbox.compile_disp(expr=el[0],),
                                         toolbox.compile_route(expr=el[1])])
            return assemble_fct, data['hof']


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
        data = { "best_ind": self.best_ind, "hof": self.hof }

        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)

        if self.individual_trees_no == 1:
            data_txt = { "best_ind": infix_str(self.best_ind), "hof": [infix_str(ind) for ind in self.hof] }
        else:
            data_txt = { "best_ind": [infix_str(self.best_ind[0]), infix_str(self.best_ind[1])],
                         "hof": [[infix_str(ind[0]), infix_str(ind[1])] for ind in self.hof],
            }

        with open(f"{file}.txt", "w") as handle:
            json.dump(data_txt, handle, indent=2, ensure_ascii=False)