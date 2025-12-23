import operator
from deap import  gp, algorithms
from src.agents.gp.gp_common import GPBase


class GP(GPBase):
    """GP Agent class"""

    def config_gp_variation_operators(self, toolbox):
        # crossover type - https://deap.readthedocs.io/en/master/api/tools.html
        # gp.cxOnePoint() Selects a random node in parent 1; Selects a random node in parent 2;Swaps the two subtrees
        # gp.cxOnePointLeafBiased(termpb=0.1) Like cxOnePoint, but with a probability of choosing leaf nodes (terminals) rather than function nodes.
        #            Useful when: You want smaller, more stable changes; You want to reduce wild tree growth (bloat); Your trees contain many terminals
        # gp.cxSemantic()
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

        # mutation type
        ##gp.mutUniform() Replaces a random subtree with a newly generated one.
        ##gp.mutNodeReplacement() Replaces one random node (function or terminal) with another of the same arity.
        ##gp.mutInsert() Inserts a new subtree above a randomly chosen node.
        ##gp.mutShrink() Replaces a subtree with one of its terminals (or smaller subtree).
        ##gp.mutEphemeral() Randomizes the value of an ephemeral constant.
        # gp.staticLimit() (wrapper) Not a mutation, but wraps a mutation to enforce depth/size limits.
        # mutSemantic
        if not hasattr(toolbox, 'expr') or not hasattr(toolbox, 'pset'):
            raise AttributeError("Toolbox not fully configured for mutation. Missing 'expr' or 'pset'.")
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=toolbox.pset)
        toolbox.decorate("mutate",
                         gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_expression_depth))

    def runGP(self, toolbox, mstats):
        print("\n--- Start GP ---")
        final_pop, logbook = algorithms.eaSimple(
                self.pop, toolbox,
                cxpb=self.cxpb, mutpb=self.mutpb,
                ngen=self.ngen,
                stats=mstats,
                halloffame=self.hof,
                verbose=True
        )
        return final_pop, logbook
