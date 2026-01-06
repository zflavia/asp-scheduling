import random
import concurrent.futures
import pickle
import traceback
import json
import numpy as np

from deap import base, creator, tools, gp
import operator
import copy
from typing import List, Tuple

from src.utils.file_handler.model_handler import ModelHandler
from src.utils.logger import Logger
from src.agents.gp.simpleTree import tree_str, infix_str, simplify_individual
from src.agents.gp.util import protected_div, protected_if, generate_random_value_for_erc, lt

class OperatorSpec:
    def __init__(self, name, optype, func):
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
        #print("op name:", self.name, self.func)
        if self.type == "mutation":
            (new_ind,) = self.func(ind1)
            return new_ind, None

        elif self.type == "crossover":
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

        self.pop_size: int = config.get('gp_population_size', 10)
        self.halloffame_size: int = config.get('gp_halloffame_size', 1)
        self.variation_probability: float = config.get('gp_population_variation', 0.95)
        #self.mutpb: float = config.get('mutation_probability', 0.3)
        self.ngen: int = config.get('gp_generations_number', 10)
        self.max_expression_depth = config.get('gp_tree_max_depth', 7)
        self.gp_tree_initial_max_depth = config.get('gp_tree_initial_max_depth', 3)
        self.simplify_frequency = config.get('gp_simplify_frequency', 10)
        self.tournament_size = config.get('gp_tournament_size', 3)
        self.np = config.get('no_parallel_processes', 1)
        self.env_config = config #for saving the solution in file

    def multi_instance_fitness(self, individual,#: gp.PrimitiveTree or list,
                           toolbox: base.Toolbox,
                           ) -> Tuple[float,]:
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
            import traceback
            traceback.print_exc()
            return (float('inf'),)

        total_combined_score = 0.0
        num_valid_instances_evaluated = 0
        self.env.current_instance_index = -1

        for inst_no in range(self.env.instances_no):
            try:
                makespan = self.env.evaluate_instance(priority_func, self.individual_trees_no)
            except Exception as e_eval:
                print("EROARE EVAL: ", e_eval)
                import traceback
                traceback.print_exc()
                makespan = float('inf')

            if makespan != float('inf'):
                total_combined_score += makespan
                num_valid_instances_evaluated += 1
            else:
                import traceback
                traceback.print_exc()
        if num_valid_instances_evaluated == 0:
            import traceback
            traceback.print_exc()
            print("Infinit!!!!!!!!!!!!")
            return (float('inf'),)

        #print(" makespan mediu individ", total_combined_score / num_valid_instances_evaluated)
        return (total_combined_score / num_valid_instances_evaluated,)

    def configure_terminals(self):
        """
        Set terminals for GP individual
        :return: pset
        """
        pass

    @classmethod
    def configure_non_terminals_and_common_primitive(cls, pset: gp.PrimitiveSetTyped):
        """
       Set non-terminals for GP individual and common primitives (constants, ERC)
       :param pset: - GP primitive set
       :return: pset
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
        stats_best_ind_obj = self.register_individual_statistic(tools)
        return tools.MultiStatistics(fitness=stats_fit,
                                       xbest_ind=stats_best_ind_obj
                                       )
    def register_individual_statistic(self, tools):
        pass

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Learn over n environment instances or n timesteps. Break depending on which condition is met first
        One learning iteration consists of collecting rollouts and training the networks

        :param total_instances: Instance limit
        :param total_timesteps: - not used, kept for compatibility with the framework
        :param intermediate_test: - not used, kept for compatibility with the framework

        """
        toolbox = self.config_gp()
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
        self.display_best_individul(self.best_ind, toolbox)

        self.save(ModelHandler.get_best_model_path(self.env_config))
        return self.best_ind.fitness.values[0]

    def display_best_individul(self, best_ind, toolbox):
        pass

    def runGP(self, toolbox, mstats):
        #implemented in subclasses
        pass


    def operator_selection_strategy_configuration(self):
        self.N_OPS = len(self.OP_SPECS)

        if self.use_aos:
            self.op_probs = np.ones(self.N_OPS) / self.N_OPS  # probabilități inițiale uniforme
            self.op_rewards = np.zeros(self.N_OPS)  # reward acumulat
            self.op_counts = np.zeros(self.N_OPS) + 1e-9  # câte ori a fost folosit fiecare operator (evităm /0)
            self.ALPHA_PM = 0.8  # “learning rate” pentru Probability Matching

        if self.use_qlearning:
            # Q-learning pe operatori
            self.Q_ops = np.zeros(self.N_OPS)  # Q-value pentru fiecare operator
            self.ALPHA_Q = 0.2  # learning rate (poți ajusta)
            self.EPSILON_Q = 0.1  # explorare (10% random)

    def update_operator_probs(self):
        avg_rewards = self.op_rewards / self.op_counts
        if np.all(avg_rewards == 0):
            self.op_probs = np.ones(self.N_OPS) / self.N_OPS
        else:
            target_probs = avg_rewards / np.sum(avg_rewards)
            uniform = np.ones(self.N_OPS) / self.N_OPS
            self.op_probs = (1 - self.ALPHA_PM) * uniform + self.ALPHA_PM * target_probs

        print("op_rewards", self.op_rewards)
        print("op_counts", self.op_counts)
        print("self.op_probs", self.op_probs)
        self.op_rewards[:] = 0.0
        self.op_counts[:] = 1e-9

    def config_gp(self):

        import inspect
        print("GPBase defined in:", inspect.getfile(GPBase))
        print("self class defined in:", inspect.getfile(self.__class__))

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
        pass

    def config_individual(self, toolbox):
        print("in base class")
        # implemented in subclasses
        pass

    def eaSimpleGP(self, population, toolbox, varpb, ngen, stats=None, halloffame=None, verbose=__debug__):

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
            # select + colonare
            #offspring = tools.selTournament(population, len(population), tournsize=3)
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for i, ind in enumerate(offspring):
                if callable(ind) or not hasattr(ind, "fitness"):
                    raise TypeError(f"Selected offspring invalid at {i}: {type(ind)} {ind}")

            # salvăm fitness-ul părintelui pentru reward
            for ind in offspring:
                if hasattr(ind, "op_id"): delattr(ind, "op_id")
                ind.parent_fitness = ind.fitness.values[0]

            # === AOS: aplicăm ORI mutație ORI crossover din aceeași listă ===
            for i in range(0, len(offspring) - 1, 2):
                ind1 = offspring[i]
                ind2 = offspring[i + 1]

                r = random.random()

                if (self.use_aos or self.use_qlearning) and r >= varpb:
                    # no operator is applied on this pair
                    continue

                # choose an operator
                if self.use_aos:
                    op_idx = np.random.choice(np.arange(self.N_OPS), p=self.op_probs)
                elif self.use_qlearning:
                    #Selectează indexul unui operator folosind o politică ε-greedy peste Q_ops.
                    if random.random() < self.EPSILON_Q:
                        # explorare: alegem un operator random
                        op_idx = random.randrange(self.N_OPS)
                    else:
                        # exploatare: alegem operatorul cu Q maxim
                        op_idx =  int(np.argmax(self.Q_ops))
                else: #no strategy to select operator
                    op_idx = np.random.choice(np.arange(self.N_OPS))

                op_spec = self.OP_SPECS[op_idx]

                if op_spec.type == "crossover":
                    # CROSSOVER
                    child1, child2 = op_spec.apply(ind1, ind2)
                    child1.op_id = op_idx
                    child2.op_id = op_idx

                    offspring[i], offspring[i + 1] = child1, child2
                    if hasattr(child1, "fitness"):
                        if hasattr(child1.fitness, "values"):
                            del child1.fitness.values
                    if hasattr(child2, "fitness"):
                        if hasattr(child2.fitness, "values"):
                            del child2.fitness.values

                else:
                    # MUTATION
                    child1, _ = op_spec.apply(ind1)
                    child2, _ = op_spec.apply(ind2)

                    child1.op_id = op_idx
                    child2.op_id = op_idx

                    offspring[i], offspring[i + 1] = child1, child2
                    if hasattr(child1, "fitness"):
                        if hasattr(child1.fitness, "values"):
                            del child1.fitness.values
                    if hasattr(child2, "fitness"):
                        if hasattr(child2.fitness, "values"):
                            del child2.fitness.values

            # =================================================================
            if gen % self.simplify_frequency == 0:
                offspring = self.simplify_population(offspring, toolbox)


            # evaluate newly created individuals
            for i, ind in enumerate(offspring):
                if not hasattr(ind, "fitness"):
                    raise TypeError(
                        f"Invalid offspring at index {i}: type={type(ind)} value={ind}"
                    )

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

                    o = ind.op_id  # indexul operatorului folosit
                    if self.use_aos:
                        self.op_rewards[o] += reward
                        self.op_counts[o] += 1.0

                    if self.use_qlearning:
                        # update Q(o) ← Q(o) + α * (r - Q(o))
                        self.Q_ops[o] = self.Q_ops[o] + self.ALPHA_Q * (reward - self.Q_ops[o])

            if self.use_aos:
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
        evaluation_type = config.get('evaluation_type', 'best')
        print("load-self.evaluation_type", evaluation_type)

        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)
            print("load() data: ", data)

        toolbox = base.Toolbox()
        from src.agents.gp.gp_alg_aos import GP_AOS
        from src.agents.gp.gp_alg_disp_route import GP_Disp_Route
        if issubclass(cls, GP_AOS):
            toolbox.register("compile", gp.compile, pset=cls.configure_terminals())
        elif issubclass(cls, GP_Disp_Route):
            pset_disp, pset_route = cls.configure_terminals()

            toolbox.register("compile_disp", gp.compile, pset=pset_disp)
            toolbox.register("compile_route", gp.compile, pset=pset_route)

        print("evaluation_type", evaluation_type)
        if evaluation_type == 'best':
            best = data['best_ind']
            from src.agents.gp.gp_alg_aos import GP_AOS
            from src.agents.gp.gp_alg_disp_route import GP_Disp_Route
            if issubclass(cls, GP_AOS):
                return toolbox.compile(expr=best), best
            elif issubclass(cls, GP_Disp_Route):
                return [toolbox.compile_disp(expr=best[0]),
                        toolbox.compile_route(expr=best[1])], best
        elif evaluation_type == 'assemble' or evaluation_type == 'assemble-test':
            assemble_fct = []
            print('hof', len(data['hof']),data['hof'])
            for el in data['hof']:
                from src.agents.gp.gp_alg_aos import GP_AOS
                from src.agents.gp.gp_alg_disp_route import GP_Disp_Route
                print(cls)
                print("GP_DS?", issubclass(cls, GP_Disp_Route))
                print("GP_AOS?", issubclass(cls, GP_AOS))
                if issubclass(cls, GP_AOS):
                    assemble_fct.append(toolbox.compile(expr=el))
                elif issubclass(cls, GP_Disp_Route):
                    assemble_fct.append([toolbox.compile_disp(expr=el[0],),
                                         toolbox.compile_route(expr=el[1])])
                print(el,assemble_fct)
            print("assemble_fct",len(assemble_fct))
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
        data = {
            "best_ind": self.best_ind,
            "hof": self.hof
        }

        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)

        if self.individual_trees_no == 1:
            data_txt = {
                "best_ind": infix_str(self.best_ind),
                "hof": [infix_str(ind) for ind in self.hof],
            }
        else:
            data_txt = {
                "best_ind": [infix_str(self.best_ind[0]), infix_str(self.best_ind[1])],
                "hof": [[infix_str(ind[0]), infix_str(ind[1])] for ind in self.hof],
            }

        with open(f"{file}.txt", "w") as handle:
            json.dump(data_txt, handle, indent=2, ensure_ascii=False)