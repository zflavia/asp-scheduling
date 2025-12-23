import random
import operator
import numpy as np
from deap import base, creator, tools, gp, algorithms
from src.agents.gp.gp_common import GPBase
from src.agents.gp.simpleTree import  simplify_individual, infix_str

class OperatorSpec:
    def __init__(self, name, optype, func):
        """
        name   : string pentru debug/log
        optype : "mutation" sau "crossover"
        func   : funcția DEAP deja înregistrată în toolbox
        """
        self.name = name
        self.type = optype
        self.func = func

    def apply(self, ind1, ind2=None):
        """
        Apelează operatorul într-un mod unificat.
        - mutation: primește 1 individ și întoarce (new_ind, None)
        - crossover: primește 2 indivizi și întoarce (child1, child2)
        """
        if self.type == "mutation":
            new_ind, = self.func(ind1)  # DEAP mutXYZ întoarce (indiv,)
            return new_ind, None
        elif self.type == "crossover":
            child1, child2 = self.func(ind1, ind2)
            return child1, child2
        else:
            raise ValueError(f"Unknown operator type: {self.type}")

class GP_AOS(GPBase):
    """GP Agent class"""

    def config_gp_variation_operators(self, toolbox):
        if not hasattr(toolbox, 'expr') or not hasattr(toolbox, 'pset'):
            raise AttributeError("Toolbox not fully configured for mutation. Missing 'expr' or 'pset'.")

        # înregistrăm operatorii de mutație individual
        toolbox.register("mut_shrink", gp.mutShrink)
        toolbox.decorate("mut_shrink",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        toolbox.register("mut_uniform", gp.mutUniform, expr=toolbox.expr, pset=toolbox.pset)
        toolbox.decorate("mut_uniform",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        toolbox.register("mut_node_replacement", gp.mutNodeReplacement, pset=toolbox.pset)
        toolbox.decorate("mut_node_replacement",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))


        toolbox.register("cx_one_point", gp.cxOnePoint)
        toolbox.decorate("cx_one_point",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        toolbox.register("cx_one_point_leaf", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.decorate("cx_one_point_leaf",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        # LISTA UNIFICATĂ DE OPERATORI (mut + cx)
        self.OP_SPECS = [
            OperatorSpec("mut_shrink", "mutation", toolbox.mut_shrink),
            OperatorSpec("mut_uniform", "mutation", toolbox.mut_uniform),
            OperatorSpec("mut_node_replacement", "mutation", toolbox.mut_node_replacement),
            OperatorSpec("cx_one_point", "crossover", toolbox.cx_one_point),
            OperatorSpec("cx_one_point_leaf", "crossover", toolbox.cx_one_point_leaf),
        ]

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
        print("self.op_probss", self.op_probs)
        self.op_rewards[:] = 0.0
        self.op_counts[:] = 1e-9

    def eaSimpleAOS(self, population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

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
        print("ngen", ngen)
        for gen in range(1, ngen + 1):
            # select + colone
            offspring = tools.selTournament(population, len(population), tournsize=3)
            offspring = list(map(toolbox.clone, offspring))

            # salvăm fitness-ul părintelui pentru reward
            for ind in offspring:
                if hasattr(ind, "op_id"):
                    delattr(ind, "op_id")
                ind.parent_fitness = ind.fitness.values[0]

            # === AOS: aplicăm ORI mutație ORI crossover din aceeași listă ===
            for i in range(0, len(offspring) - 1, 2):
                ind1 = offspring[i]
                ind2 = offspring[i + 1]

                r = random.random()
                if r >= cxpb + mutpb:
                    # nu aplicăm niciun operator GP pe această pereche
                    continue

                # alegem un operator din AOS (mut sau cx)
                if self.use_aos:
                    op_idx = np.random.choice(np.arange(self.N_OPS), p=self.op_probs)
                if self.use_qlearning:
                    #Selectează indexul unui operator folosind o politică ε-greedy peste Q_ops.
                    if random.random() < self.EPSILON_Q:
                        # explorare: alegem un operator random
                        op_idx = random.randrange(self.N_OPS)
                    else:
                        # exploatare: alegem operatorul cu Q maxim
                        op_idx =  int(np.argmax(self.Q_ops))
                op_spec = self.OP_SPECS[op_idx]

                if op_spec.type == "crossover":
                    # CROSSOVER - folosește ambii părinți
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
                    # MUTATION - aplicăm pe fiecare în parte, cu același operator
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

            offspring  = [simplify_individual(ind, toolbox.pset) for ind in offspring]

            # hall of fame + log
            if halloffame is not None:
                halloffame.update(offspring)

            population[:] = toolbox.select(offspring, len(population))

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    def runGP(self, toolbox, mstats):
        print("\n--- Start GP ---")
        final_pop, logbook = self.eaSimpleAOS(  # algorithms.eaSimple(
            self.pop, toolbox,
            cxpb=self.cxpb, mutpb=self.mutpb,
            ngen=self.ngen,
            stats=mstats,
            halloffame=self.hof,
            verbose=True
        )

        return final_pop, logbook