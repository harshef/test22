from nsga2.evolution import Evolution
from nsga2.problems.zdt import ZDT
from nsga2.problems.zdt.zdt3_definitions import ZDT3Definitions

from metrics.problems.zdt import ZDT3Metrics
from plotter import Plotter

collected_metrics = {}
def collect_metrics(population, generation_num):
    pareto_front = population.fronts[0]
    metrics = ZDT3Metrics()
    hv = metrics.HV(pareto_front)
    hvr = metrics.HVR(pareto_front)
    collected_metrics[generation_num] = hv, hvr

def Select(population,sil_sco,DunnIndex,nsol, new_di, new_ss,generation,self_population,Zdt_definitions,PPlotter,PProblem,EEvolution,Final_label,label):
    if (Zdt_definitions is None) and (PPlotter is None) and (PProblem is None) and (EEvolution is None):

        zdt_definitions = ZDT3Definitions()
        plotter = Plotter(zdt_definitions)
        problem = ZDT(zdt_definitions)
        evolution = Evolution(problem, 1, len(population))
        evolution.register_on_new_generation(plotter.plot_population_best_front)
        evolution.register_on_new_generation(collect_metrics)
    else:
        zdt_definitions=Zdt_definitions
        plotter=PPlotter
        problem=PProblem
        evolution=EEvolution

    print "++++++++++"
    new_pop,objectives,self_population,Final_label= evolution.evolve(population,sil_sco,DunnIndex,nsol,new_di,new_ss,generation,self_population,Final_label,label)

    return new_pop,objectives,self_population,zdt_definitions,plotter,problem,evolution,Final_label
