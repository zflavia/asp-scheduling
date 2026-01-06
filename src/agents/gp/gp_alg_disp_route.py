from src.agents.gp.gp_common import GPBase, OperatorSpec
from deap import base, creator, tools, gp, algorithms
from src.utils.logger import Logger
import operator
import random
from src.agents.gp.simpleTree import  simplify_individual, infix_str, tree_str

class GP_Disp_Route(GPBase):
    """GP Agent class"""

    def __init__(self, env, config: dict, logger: Logger = None, metadata=None):
        super().__init__(env, config, logger, metadata)
        self.individual_trees_no = 2

    @classmethod
    def configure_terminals(cls) -> base.Toolbox:
        """
        Describe the GP individual
        :return: configured DEAP object
        """
        # Dispatch features:
        pset_disp = gp.PrimitiveSetTyped("DISP", [float] *6 , float)

        # 1. Operation: Mean processing time: Estimates operation duration.
        pset_disp.renameArguments(ARG0='O_MeanPT')
        # 2. Operation: Minimum processing time: Highlights the quickest possible execution time
        pset_disp.renameArguments(ARG1='O_MinPT')
        # 3. Operation:  Ratio of machines that are eligible for Oij to total machine number
        pset_disp.renameArguments(ARG2='O_Flex')
        # 4. Operation: number of operations from the current operation to root operation
        pset_disp.renameArguments(ARG3="O_Path_OpNo")
        # 5. Operation:  length of the execution path from the current operation to root operation (computed using min
        # execution time from all machine alternatives)
        pset_disp.renameArguments(ARG4="O_Path_MinLen")
        # 6. Operation: current makespan - operation.release_time
        pset_disp.renameArguments(ARG5="O_WT")

        # Routing features:
        pset_route = gp.PrimitiveSetTyped("ROUTE", [float]*8, float)
        # 4. Edge (op, machine): Processing time p_{ik}  of operation i on machine k
        pset_route.renameArguments(ARG0='E_PT')
        # 7. Machine: Last operation completion time t_{last}: Determines machine availability.
        pset_route.renameArguments(ARG1='M_RT')
        # 8. Machine: Number of operations (unscheduled)  that can be executed on M / total number of operations (unscheduled)
        pset_route.renameArguments(ARG2='M_OP')
        # 9. Machine: Utilization percentage: T_{used}/t_{last}: Indicates machine efficiency.
        pset_route.renameArguments(ARG3='M_UT')
        # 13. Machine: queue length (op. no in queue)
        pset_route.renameArguments(ARG4="M_QL")
        # 14. Machine: queue duration (op. duration in queue)
        pset_route.renameArguments(ARG5="M_QD")
        # Machine: CT_append
        pset_route.renameArguments(ARG6="M_CT_A")
        # Machine: CT_backward
        pset_route.renameArguments(ARG7="M_CT_B")

        cls.configure_non_terminals_and_common_primitive(pset_disp)
        cls.configure_non_terminals_and_common_primitive(pset_route)

        return pset_disp, pset_route


    def config_individual(self, toolbox):
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox.pset_dispatch, toolbox.pset_route  = self.configure_terminals()

        #initialisation individual
        toolbox.register("expr_disp", gp.genHalfAndHalf, pset=toolbox.pset_dispatch, min_=1, max_=self.gp_tree_initial_max_depth) # trebuie in config param
        toolbox.register("expr_route", gp.genHalfAndHalf, pset=toolbox.pset_route, min_=1, max_=self.gp_tree_initial_max_depth)
        #register individual
        toolbox.register("tree_disp", tools.initIterate, gp.PrimitiveTree, toolbox.expr_disp)
        toolbox.register("tree_route", tools.initIterate, gp.PrimitiveTree, toolbox.expr_route)
        def init_individual():
            return creator.Individual([toolbox.tree_disp(), toolbox.tree_route()])
        toolbox.register("individual", init_individual)

        # Compile individual
        toolbox.register("compile_disp", gp.compile, pset=toolbox.pset_dispatch)
        toolbox.register("compile_route", gp.compile, pset=toolbox.pset_route)

        # Evaluate individual
        toolbox.register("evaluate", self.multi_instance_fitness, toolbox=toolbox)

        return toolbox


    def config_gp_variation_operators(self, toolbox):
        def ind_height(ind):
            return max(ind[0].height, ind[1].height)

        def mut_two_trees(mut_disp, mut_route, ind):
            (ind[0],) = mut_disp(ind[0])
            (ind[1],) = mut_route(ind[1])
            return (ind,)

        def cx_two_trees_one_point(ind1, ind2):
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
            ind1[1], ind2[1] = gp.cxOnePoint(ind1[1], ind2[1])
            return ind1, ind2

        def cx_two_trees_one_point_leaf(ind1, ind2):
            ind1[0], ind2[0] = gp.cxOnePointLeafBiased(ind1[0], ind2[0], termpb = 0.1)
            ind1[1], ind2[1] = gp.cxOnePointLeafBiased(ind1[1], ind2[1], termpb = 0.1)
            return ind1, ind2

        def mut_uniform_2t(ind):
            "This is a fix for the mutUniform, it has a problem with multiple handles used by the other mutations types"
            (t0,) = gp.mutUniform(ind[0], expr=toolbox.expr_disp, pset=toolbox.pset_dispatch)
            (t1,) = gp.mutUniform(ind[1],  expr=toolbox.expr_route, pset=toolbox.pset_route)
            ind[0], ind[1] = t0, t1
            return (ind,)

        # register mutation operators
        toolbox.register("mut_shrink_disp", gp.mutShrink)
        toolbox.register("mut_shrink_route", gp.mutShrink)
        toolbox.register("mut_shrink_2t", mut_two_trees, toolbox.mut_shrink_disp, toolbox.mut_shrink_route)
        toolbox.decorate("mut_shrink_2t", gp.staticLimit(key=ind_height, max_value=self.max_expression_depth))

        toolbox.register("mut_uniform_2t", mut_uniform_2t)
        toolbox.decorate("mut_uniform_2t", gp.staticLimit(key=ind_height, max_value=self.max_expression_depth))

        toolbox.register("mut_node_replacement_disp", gp.mutNodeReplacement, #expr=toolbox.expr_disp,
                         pset=toolbox.pset_dispatch)
        toolbox.register("mut_node_replacement_route", gp.mutNodeReplacement, #expr=toolbox.expr_route,
                         pset=toolbox.pset_route)
        toolbox.register("mut_node_replacement_2t",mut_two_trees,toolbox.mut_node_replacement_disp,toolbox.mut_node_replacement_route)
        toolbox.decorate("mut_node_replacement_2t", gp.staticLimit(key=ind_height, max_value=self.max_expression_depth))

        # register crossover operators
        toolbox.register("cx_one_point_2t", cx_two_trees_one_point)
        toolbox.decorate("cx_one_point_2t", gp.staticLimit(key=ind_height, max_value=self.max_expression_depth))

        toolbox.register("cx_one_point_leaf_2t", cx_two_trees_one_point_leaf)
        toolbox.decorate("cx_one_point_leaf_2t", gp.staticLimit(key=ind_height, max_value=self.max_expression_depth))

        self.OP_SPECS = [
            OperatorSpec("mut_shrink", "mutation", toolbox.mut_shrink_2t),
            OperatorSpec("mut_uniform", "mutation", toolbox.mut_uniform_2t),
            OperatorSpec("mut_node_replacement", "mutation", toolbox.mut_node_replacement_2t),
            OperatorSpec("cx_one_point", "crossover", toolbox.cx_one_point_2t),
            OperatorSpec("cx_one_point_leaf", "crossover", toolbox.cx_one_point_leaf_2t),
        ]

        return self.OP_SPECS

    def simplify_population(self, offspring, toolbox):
        for ind in offspring:
            ind[0] = simplify_individual(ind[0], toolbox.pset_dispatch)
            ind[1] = simplify_individual(ind[1], toolbox.pset_route)

        return offspring

    def display_best_individul(self, best_ind, toolbox):
        print("Best individual:", tree_str(best_ind[0]), tree_str(best_ind[1]))
        print(" dispatch:", infix_str(best_ind[0]))  # vezi formula matematică
        print(" route:", infix_str(best_ind[1]))  # vezi formula matematică

        simp = simplify_individual(best_ind[0], toolbox.pset_dispatch)
        print("Simplified dispatch:", infix_str(simp))
        simp = simplify_individual(best_ind[1], toolbox.pset_route)
        print("Simplified route:", infix_str(simp))

    def register_individual_statistic(self, tools):
        stats_best_ind_obj = tools.Statistics(key=lambda ind: ind)

        def disp_expr(inds):
            valid = [i for i in inds if i.fitness.valid]
            if not valid:
                return None
            best = min(valid, key=lambda i: i.fitness.values[0])
            return str(best[0])

        def route_expr(inds):
            valid = [i for i in inds if i.fitness.valid]
            if not valid:
                return None
            best = min(valid, key=lambda i: i.fitness.values[0])
            return str(best[1])

        stats_best_ind_obj.register("disp_expr", disp_expr)
        stats_best_ind_obj.register("route_expr", route_expr)
        return stats_best_ind_obj
