import random
import operator
import numpy as np
from deap import base, creator, tools, gp, algorithms
from src.agents.gp.gp_common import GPBase, OperatorSpec
from src.utils.logger import Logger
from src.agents.gp.simpleTree import  simplify_individual, infix_str, tree_str

class GP_AOS(GPBase):
    """GP Agent class"""

    def __init__(self, env, config: dict, logger: Logger = None, metadata=None):
        super().__init__(env, config, logger, metadata)
        self.individual_trees_no = 1

    @classmethod
    def configure_terminals(cls):
        """
        Set terminals for GP individual
        :return: pset
        """
        #pset = gp.PrimitiveSetTyped("GP_pair", [float]*16, float)  # 9
        pset = gp.PrimitiveSetTyped("GP_pair", [float] * 5, float)  # 9

        pset.renameArguments(ARG0='E_PT')
        pset.renameArguments(ARG1='E_PT_maxPT')
        pset.renameArguments(ARG2='E_PT_maxMPT')
        pset.renameArguments(ARG3="M_CT_A")
        pset.renameArguments(ARG4="M_CT_B")


        # Score function arguments:
        # #1. Operation: Mean processing time: Estimates operation duration.
        # pset.renameArguments(ARG0='O_MeanPT')
        # #2. Operation: Minimum processing time: Highlights the quickest possible execution time
        # pset.renameArguments(ARG1='O_MinPT')
        # #3. Operation:  Ratio of machines that are eligible for Oij to total machine number
        # pset.renameArguments(ARG2='O_Flex')
        #4. Edge (op, machine): Processing time p_{ik}  of operation i on machine k
        # pset.renameArguments(ARG3='E_PT')
        # #5. Edge (op, machine): Ratio of p_{ik} to the maximum processing time of p_{il}  l=1,M_i  (M_i= total number
        # # of machines on which op i can be executed)
        # pset.renameArguments(ARG4='E_PT_maxPT')
        # #6. Edge (op, machine): Ratio of p_{ik} to the maximum processing time of p_{lk}  l=1,N _k (N_k= total number
        # # of operations that can be executed on machine k)
        # pset.renameArguments(ARG5='E_PT_maxMPT')
        # #7. Machine: Last operation completion time t_{last}: Determines machine availability.
        # pset.renameArguments(ARG6='M_RT')
        # #8. Machine: Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
        # pset.renameArguments(ARG7='M_OP')
        # #9. Machine: Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
        # pset.renameArguments(ARG8='M_UT')
        # #10. Operation: number of operations from the current operation to root operation
        # pset.renameArguments(ARG9="O_Path_OpNo")
        # #11. Operation:  lenght of the execution path from the current operation to root operation (computed using min
        # # execution time from all machine alternatives)
        # pset.renameArguments(ARG10="O_Path_MinLen")
        # #12. Operation: current makespan - operation.release_time
        # pset.renameArguments(ARG11="O_WT")
        # #13. Machine: queue length (op. no in queue)
        # pset.renameArguments(ARG12="M_QL")
        # #14. Machine: queue duration (op. duration in queue)
        # pset.renameArguments(ARG13="M_QD")
        #15. Machine: CT_append
        # pset.renameArguments(ARG6="M_CT_A")
        # #16. Machine: CT_backward
        # pset.renameArguments(ARG7="M_CT_B")

        cls.configure_non_terminals_and_common_primitive(pset)
        return pset

    def config_individual(self, toolbox):
        toolbox.pset = self.configure_terminals()

        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=toolbox.pset)

        #initialisation individual
        toolbox.register("expr", gp.genHalfAndHalf, pset=toolbox.pset, min_=1, max_=self.gp_tree_initial_max_depth)  # (1,5) adaincimea in populatia initiala
        # register tree individual
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

        # Compile individual
        toolbox.register("compile", gp.compile, pset=toolbox.pset)

        # register fitness function
        toolbox.register("evaluate", self.multi_instance_fitness, toolbox=toolbox)

        return toolbox


    def config_gp_variation_operators(self, toolbox):
        if not hasattr(toolbox, 'expr') or not hasattr(toolbox, 'pset'):
            raise AttributeError("Toolbox not fully configured for mutation. Missing 'expr' or 'pset'.")

        # register mutation operators
        toolbox.register("mut_shrink", gp.mutShrink)
        toolbox.decorate("mut_shrink",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        toolbox.register("mut_uniform", gp.mutUniform, expr=toolbox.expr, pset=toolbox.pset)
        toolbox.decorate("mut_uniform",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        toolbox.register("mut_node_replacement", gp.mutNodeReplacement, pset=toolbox.pset)
        toolbox.decorate("mut_node_replacement",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        # register crossover operators
        toolbox.register("cx_one_point", gp.cxOnePoint)
        toolbox.decorate("cx_one_point",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        toolbox.register("cx_one_point_leaf", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.decorate("cx_one_point_leaf",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        # Unified operator list (mut + cx)
        self.OP_SPECS = [
            OperatorSpec("mut_shrink", "mutation", toolbox.mut_shrink),
            OperatorSpec("mut_uniform", "mutation", toolbox.mut_uniform),
            OperatorSpec("mut_node_replacement", "mutation", toolbox.mut_node_replacement),
            OperatorSpec("cx_one_point", "crossover", toolbox.cx_one_point),
            OperatorSpec("cx_one_point_leaf", "crossover", toolbox.cx_one_point_leaf),
        ]

        return self.OP_SPECS

        # self.N_OPS = len(self.OP_SPECS)
        #
        # if self.use_aos:
        #     self.op_probs = np.ones(self.N_OPS) / self.N_OPS  # probabilități inițiale uniforme
        #     self.op_rewards = np.zeros(self.N_OPS)  # reward acumulat
        #     self.op_counts = np.zeros(self.N_OPS) + 1e-9  # câte ori a fost folosit fiecare operator (evităm /0)
        #     self.ALPHA_PM = 0.8  # “learning rate” pentru Probability Matching
        #
        # if self.use_qlearning:
        #     # Q-learning pe operatori
        #     self.Q_ops = np.zeros(self.N_OPS)  # Q-value pentru fiecare operator
        #     self.ALPHA_Q = 0.2  # learning rate (poți ajusta)
        #     self.EPSILON_Q = 0.1  # explorare (10% random)

    def simplify_population(self, offspring, toolbox):
        for ind in offspring:
            new_tree = simplify_individual(ind, toolbox.pset)
            ind.clear()
            ind.extend(new_tree)
        return offspring

    def display_best_individul(self, best_ind, toolbox):
        print("Best individual:", tree_str(best_ind))
        print(infix_str(best_ind))  # vezi formula matematică

        simp = simplify_individual(best_ind, toolbox.pset)
        print("Simplified:")
        print(infix_str(simp))

    def register_individual_statistic(self, tools):
        stats_best_ind_obj = tools.Statistics(key=lambda ind: ind)
        stats_best_ind_obj.register("best", lambda pop_list: min(pop_list, key=lambda
            ind: ind.fitness.values[0] if ind.fitness.valid else float('inf')))
        return stats_best_ind_obj