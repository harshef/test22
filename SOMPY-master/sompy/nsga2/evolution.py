"""Module with main parts of NSGA-II algorithm.
Contains main loop"""
from nsga2 import individual
from nsga2.utils import NSGA2Utils
from nsga2.population import Population



class Evolution(object):

    def __init__(self, problem, num_of_generations, num_of_individuals):
        self.utils = NSGA2Utils(problem, num_of_individuals)

        self.population = None

        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def register_on_new_generation(self, fun):
        self.on_generation_finished.append(fun)
        
    def evolve(self,solution,sil_sco,DunnIndex,new_solution,new_di,new_ss,generation,self_population,Final_label,label):

        if self_population is None:

            self.population = self.utils.create_initial_population(sil_sco,DunnIndex,solution)
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:

                self.utils.calculate_crowding_distance(front)

        else:
            print "SelfPOP", self_population, len(self_population), type(self_population), self_population.fronts
            self.population=self_population
        nsol =[]
        objective=[]
        cc = 0
        children = self.utils.create_children(new_solution,new_di,new_ss)

        self.population.extend(children)

        self.utils.fast_nondominated_sort(self.population)

        new_population = Population()
        front_num = 0
        while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            new_population.extend(self.population.fronts[front_num])
            front_num += 1

        sorted(self.population.fronts[front_num], cmp=self.utils.crowding_operator)
        new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
        returned_population = self.population
        self.population=new_population

        for _ in new_population:
            x = (getattr(_, 'features'))
            obj = (getattr(_, 'objectives'))
            nsol.insert(cc, x)
            objective.insert(cc, obj)
            cc += 1
        for fun in self.on_generation_finished:
            print "Fun",fun,returned_population,self.on_generation_finished
            fun(returned_population, generation)
            Final_label_old=list(Final_label)

        ############################ Label Updation############################
        new_solution = list(new_solution) #Features of children chromosome
        solution = list(solution) #List of all original population i.e, before NSGA
        nsol # list of all new population features
        for k in range(len(solution)):
            y=list(solution[k])
            if y in nsol:
                get_index= nsol.index(y)
                Final_label[get_index]=Final_label_old[k]
            elif (new_solution in nsol) and (y not in nsol):
                get_index = nsol.index(new_solution)
                Final_label[get_index] = label
        #########################################################################

        return nsol,objective,self.population,Final_label
