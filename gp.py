"""
using code by © moshe sipper, www.moshesipper.com
author: Jan Mayer
"""
from random import random, randint, seed, choice
from math import sin, cos, pi, exp, sqrt
import numpy as np
from statistics import mean
from copy import deepcopy
import plotting as pl
import traceback

class Parameters:
    """
    Purpose if this class is only to keep current gp parameters.
    """
    gp_rules = None

class GPTree:
    """
    This class represents a tree in genetic programing.
    """
    def __init__(self, data=None, left=None, right=None):
        """Constructor

        Parameters
        ----------
        data : a function or a terminal
            data at the node
        left : GPTree
            left subtree
        right : GPTree
            right subtree

        """
        self.data  = data
        self.left  = left
        self.right = right

    def _isnumber(self, x):
        """A simple function used to determine if a string represents a number/

        Parameters
        ----------
        x : str
            input string

        Returns
        -------
        bool
            True if x represents a number, False otherwise

        """
        try:
            float(x)
            return True
        except:
            return False

    def to_str_simplified(self):
        """method converts the tree to string and also simplifies expressions
        like ((2 + 1) * x) to (3 * x)

        Returns
        -------
        str
            string representation of the tree

        """
        if self.data in Parameters.gp_rules['binary_functions']:
            s1 = self.left.to_str_simplified()
            s2 = self.right.to_str_simplified()
            if self._isnumber(s1) and self._isnumber(s2):
                return str(self.data(float(s1), float(s2)))
            else:
                return f'({s1} {Parameters.gp_rules["func_names"][self.data]} {s2})'
        elif self.data in Parameters.gp_rules['unary_functions']:
            s1 = self.left.to_str_simplified()
            if self._isnumber(s1):
                return str(self.data(float(s1)))
            else:
                return f'{Parameters.gp_rules["func_names"][self.data]}({s1})'
        else:
            return self.node_label()

    def node_label(self):
        """Method returns name of node's data.

        Returns
        -------
        str
            name od node's data

        """
        if (self.data in (Parameters.gp_rules['binary_functions'] + Parameters.gp_rules['unary_functions'])):
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix = ""):
        """Prints a tree.

        Parameters
        ----------
        prefix : str
            Prefix before output, used in recursion to make the output look like
            a tree.

        """
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right and self.data not in Parameters.gp_rules['unary_functions']: self.right.print_tree(prefix + "   ")

    def compute_tree(self, inputs):
        """Compute the value of the tree for given inputs.

        Parameters
        ----------
        inputs : (d,) np.array
            a d-dimensional input

        Returns
        -------
        type depends on return type of the function in the root, usually python float
            value of the tree

        """
        if (self.data in Parameters.gp_rules['binary_functions']):
            return self.data(self.left.compute_tree(inputs), self.right.compute_tree(inputs))
        elif (self.data in Parameters.gp_rules['unary_functions']):
            return self.data(self.left.compute_tree(inputs))
        elif self.data in Parameters.gp_rules['inputs']:
            return inputs[Parameters.gp_rules['inputs'].index(self.data)]
        else: return self.data

    def random_tree(self, grow, max_depth, depth=False):
        """Make the tree a new, random one.

        Parameters
        ----------
        grow : bool
            strategy to use
        max_depth : int
            maximal depth of the tree
        depth : int
            current dept, used for recursion

        """
        if depth < Parameters.gp_rules['min_depth'] or (depth < max_depth and not grow):
            self.data = choice(Parameters.gp_rules['unary_functions'] + Parameters.gp_rules['binary_functions'])
        elif depth >= max_depth:
            self.data = choice(Parameters.gp_rules['terminals'])
        else: # intermediate depth, grow
            if random () > 0.5:
                self.data = choice(Parameters.gp_rules['terminals'])
            else:
                self.data = choice(Parameters.gp_rules['unary_functions'] + Parameters.gp_rules['binary_functions'])
        if self.data in Parameters.gp_rules['binary_functions']:
            self.left = GPTree(Parameters.gp_rules)
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree(Parameters.gp_rules)
            self.right.random_tree(grow, max_depth, depth=depth + 1)
        elif self.data in Parameters.gp_rules['unary_functions']:
            self.left = GPTree(Parameters.gp_rules)
            self.left.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        """Mutates the tree by selecting random subtree and replacing it with a
        random subtree with depth 2.

        """
        if random() < Parameters.gp_rules['prob_mutation']:
            self.random_tree(grow=True, max_depth=2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation()

    def size(self):
        """Count the nodes in the tree.

        Returns
        -------
        int
            number of nodes in the tree

        """
        if self.data in Parameters.gp_rules['terminals']: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):
        """Returns a copy if the tree

        """
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):
        """Go through a tree and either return subtree at count-th node or add
        second tree there.

        Parameters
        ----------
        count : [int] (not sure why)
            number of nodes left to visit
        second : GPTree
            tree to insert, None if we just want to return the subtree

        Returns
        -------
        GPTree
            subtree or None if subtree is to be added

        """
        count[0] -= 1
        if count[0] <= 1:
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:
            ret = None
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):
        """perform crossover

        Parameters
        ----------
        other : GPTree
            tree to be crossovered with

        """
        if random() < Parameters.gp_rules['xo_rate']:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)

class GeneticProgram():
    """Class representing a genetic program.

    Attributes
    ----------
    population : (n,) np.array, dtype=GPTree
        current population of solutions
    fitnesses : (n,) np.array
        array of fitnesses of all solutions

    """

    def __init__(self, dataset):
        self.population = np.empty((Parameters.gp_rules['pop_size'],), dtype=GPTree)
        self.fitnesses = np.zeros((Parameters.gp_rules['pop_size']))
        self.dataset = dataset
        self.init_population()
        self.test_cases_evaluated = 0

    def init_population(self):
        """Create random population

        """
        idx = 0
        for _ in range(Parameters.gp_rules['pop_size'] // 2):
            t1 = GPTree(Parameters.gp_rules)
            t1.random_tree(grow=True, max_depth=randint(Parameters.gp_rules['min_depth'], Parameters.gp_rules['max_depth']))
            t2 = GPTree(Parameters.gp_rules)
            t2.random_tree(grow=False, max_depth=randint(Parameters.gp_rules['min_depth'], Parameters.gp_rules['max_depth']))
            self.population[idx] = t1
            self.population[idx + 1] = t2
            idx += 2
        if idx < Parameters.gp_rules['pop_size']: # add one more for odd pop_size
            t1 = GPTree(Parameters.gp_rules)
            t1.random_tree(grow=True, max_depth=randint(Parameters.gp_rules['min_depth'], Parameters.gp_rules['max_depth']))
            self.population[idx] = t1

    def fitness(self, individual, test_cases=None):
        """fitness function of the program

        Parameters
        ----------
        individual : GPTree
            individual we want to evaluate
        test_cases : array of ints
            indexes of test_cases to use of evaluation, set to None to use all tests

        Returns
        -------
        float
            mean absolute error on all tests

        """
        if test_cases is None:
            test_cases = range(len(self.dataset))
        self.test_cases_evaluated += len(test_cases)
        try:
            return mean([abs(individual.compute_tree(ds[:-1]) - ds[-1]) for ds in self.dataset[test_cases]])
        except Exception as e:
            print('exception during evaluation, shouldnt be here :(')
            print(traceback.format_exc())
            return np.inf

    def selection(self):
        """Selects an andividual using tournament selection.

        Returns
        -------
        GPTree
            copy of the winner of the tournament

        """
        tournament = np.random.randint(len(self.population), size=Parameters.gp_rules['tournament_size'], dtype=np.int32) # select tournament contenders
        tournament_fitnesses = self.fitnesses[tournament]
        return deepcopy(self.population[tournament[np.argmin(tournament_fitnesses)]])

    def evaluate_population(self, test_cases=None):
        """computes and saves fitness of the population

        Parameters
        ----------
        dataset : (m, d) np.array
            see function fitness for explanation
        test_cases : array of ints
            indexes of test_cases to use of evaluation, set to None to use all tests

        """
        for i in range(len(self.fitnesses)):
            self.fitnesses[i] = self.fitness(self.population[i], test_cases=test_cases)

    def evolve_population(self):
        """Evolves the current population using mutation and crossover

        """
        for i in range(Parameters.gp_rules['pop_size']):
            parent1 = self.selection()
            parent2 = self.selection()
            parent1.crossover(parent2)
            parent1.mutation()
            self.population[i] = deepcopy(parent1)

    def run_evolution(self, fp_manager=None, verbose=False):
        """perform an evolution run

        Parameters
        ----------
        verbose : bool
            set to True if you want the method to print progress

        Returns
        -------
        dict
            stats and results of the run

        """
        # init
        self.init_population()
        best_of_run = None
        best_of_run_f = np.inf
        best_of_run_gen = 0
        end_criteria_met = False

        avg_fitnesses = []
        best_of_gen_fitnesses = []
        best_of_run_fitnesses = []
        avg_sizes = []
        test_cases_evaluations = []
        best_of_run_exact = np.inf

        self.test_cases_evaluated = 0

        # start
        cur_gen = 0
        while not end_criteria_met:
            cur_gen += 1
            if verbose and cur_gen % 100 == 0:
                print('current generation {}, {:.2e}'.format(cur_gen, self.test_cases_evaluated))
            if fp_manager is not None:
                test_cases = fp_manager.get_best_predictor().test_cases
                fp_manager.next_generation(generation=cur_gen)
            else:
                test_cases = None
            self.evolve_population()
            self.evaluate_population(test_cases=test_cases)

            best_of_gen = np.argmin(self.fitnesses)
            avg_fitnesses.append(mean(self.fitnesses))
            best_of_gen_fitnesses.append(self.fitness(self.population[best_of_gen]))
            avg_sizes.append(mean([t.size() for t in self.population]))
            test_cases_evaluations.append(self.test_cases_evaluated)

            if self.fitnesses[best_of_gen] < best_of_run_f:
                best_of_run_f = self.fitnesses[best_of_gen]
                best_of_run_exact = self.fitness(self.population[best_of_gen])
                best_of_run_gen = cur_gen
                best_of_run = deepcopy(self.population[best_of_gen])

            best_of_run_fitnesses.append(best_of_run_exact)

            end_criteria_met |= cur_gen > Parameters.gp_rules['generations'] or best_of_run_exact < 0.1 or self.test_cases_evaluated > Parameters.gp_rules['evaluations_limit']
        return {'best' : best_of_run,
                'generations' : cur_gen + 1,
                'avg_fitnesses' : np.array(avg_fitnesses),
                'best_of_gen_fitnesses' : np.array(best_of_gen_fitnesses),
                'best_of_run_fitnesses' : np.array(best_of_run_fitnesses),
                'best_f' : best_of_run_f,
                'avg_sizes' : np.array(avg_sizes),
                'best_of_run_exact_fitness' : self.fitness(best_of_run),
                'test_cases_evaluations' : np.array(test_cases_evaluations),
                'used_test_cases' : test_cases}

class FitnessPredictor():

    def __init__(self, number_of_tests, size, test_cases=None):
        self.number_of_tests = number_of_tests
        self.test_cases = test_cases
        self.size = size
        if test_cases is None:
            self.random_predictor()

    def random_predictor(self):
        self.test_cases = np.random.randint(self.number_of_tests, size=self.size)

    def __str__(self):
        return str(self.test_cases)

class FitnessPredictorManager():

    def __init__(self, owner, dataset_size):
        self.owner = owner
        self.dataset_size = dataset_size

    def get_best_predictor(self):
        raise NotImplementedError()

    def next_generation(self, **args):
        raise NotImplementedError()

class EvolvingFitnessPredictor(FitnessPredictor):

    def __init__(self, number_of_tests, size, prob_mutation, prob_xo, test_cases=None):
        super().__init__(number_of_tests, size, test_cases=test_cases)
        self.prob_xo = prob_xo
        self.prob_mutation = prob_mutation

    def mutate(self):
        for i in range(self.size):
            if random() < self.prob_mutation:
                self.test_cases[i] = randint(0, self.number_of_tests - 1)

    def crossover(self, other):
        if self.size != other.size:
            raise ValueError('predictors must have same size')
        if random() < self.prob_xo:
            xo_point = randint(0, self.size - 1)
            self.test_cases[:xo_point] = other.test_cases[:xo_point]


class SLFitnessPredictorManager(FitnessPredictorManager):

    def __init__(self, owner, dataset_size, predictors_pop_size=8, predictors_size=8, trainers_pop_size=10, prob_mutation=0.1, prob_xo=0.5):
        super().__init__(owner, dataset_size)
        self.predictors_pop_size = predictors_pop_size
        self.predictors_size = predictors_size
        self.trainers_pop_size = trainers_pop_size
        self.predictors_pop = np.array([EvolvingFitnessPredictor(dataset_size, predictors_size, prob_mutation, prob_xo) \
                                        for _ in range(predictors_pop_size)])

        self.trainers_pop = np.empty(shape=trainers_pop_size, dtype=GPTree)
        for i in range(trainers_pop_size):
            self.trainers_pop[i] = deepcopy(choice(owner.population))
        self.trainers_fitnesses = np.array([self.trainer_fitness(t) for t in self.trainers_pop])
        self.trainers_exact_fitnesses = np.array([self.owner.fitness(t) for t in self.trainers_pop])

        self.pred_fitnesses = np.zeros_like(self.predictors_pop)
        self.evaluate_predictors()
        self.best_predictor = None

    def predictor_fitness(self, predictor):
        error_sum = 0
        for i, t in enumerate(self.trainers_pop):
            error_sum += abs(self.trainers_exact_fitnesses[i] - self.owner.fitness(t, test_cases=predictor.test_cases))
        return error_sum / self.trainers_pop_size

    def evaluate_predictors(self):
        for i in range(len(self.pred_fitnesses)):
            self.pred_fitnesses[i] = self.predictor_fitness(self.predictors_pop[i])

    def get_best_predictor(self):
        return self.predictors_pop[np.argmin(self.pred_fitnesses)]

    def next_generation(self, **args):
        if args['generation'] % 20 == 0:
            self.add_new_trainer()
            self.evolve_predictors()
            self.evaluate_predictors()

    def evolve_predictors(self):
        for i in range(len(self.predictors_pop)):
            p1 = self.selection()
            p2 = self.selection()
            p1.crossover(p2)
            p1.mutate()
            self.predictors_pop[i] = deepcopy(p1)

    def selection(self):
        tournament = np.random.randint(self.predictors_pop_size, size=3, dtype=np.int32)
        tournament_fitnesses = self.pred_fitnesses[tournament]
        return deepcopy(self.predictors_pop[tournament[np.argmin(tournament_fitnesses)]])

    def trainer_fitness(self, t):
        return np.std([self.owner.fitness(t, test_cases=p.test_cases) for p in self.predictors_pop])

    def add_new_trainer(self):
        all_fitnesses = [self.trainer_fitness(t) for t in self.owner.population]
        cur_worst = np.argmin(self.trainer_fitness)
        new = np.argmax(all_fitnesses)
        self.trainers_pop[cur_worst] = deepcopy(self.owner.population[new])
        self.trainers_fitnesses[cur_worst] = all_fitnesses[new]
        self.trainers_exact_fitnesses[cur_worst] = self.owner.fitness(self.trainers_pop[cur_worst])


class RandomFitnessPredictorManager(FitnessPredictorManager):
    """A dummy random fitness predictor manager used for testing
    """

    def __init__(self, owner, dataset_size):
        super().__init__(owner, dataset_size)

    def get_best_predictor(self):
        return FitnessPredictor(self.dataset_size, 10)

    def next_generation(self, **args):
        pass

if __name__== "__main__":
    fp = FitnessPredictor(10, 5, 0.1, 1.0)
    fp2 = FitnessPredictor(10, 5, 0.1, 1.0)
    print(fp)
    print(fp2)
    fp.crossover(fp2)
    print(fp)
