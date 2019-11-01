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
            if random() > 0.5:
                self.data = choice(Parameters.gp_rules['terminals'])
            else:
                self.data = 'x'
        else: # intermediate depth, grow
            if random () > 0.5:
                self.data = choice(Parameters.gp_rules['terminals'])
            else:
                self.data = choice(Parameters.gp_rules['unary_functions'] + Parameters.gp_rules['binary_functions'])
        if self.data in Parameters.gp_rules['binary_functions']:
            self.left = GPTree(Parameters.gp_rules)
            self.left.random_tree(grow, max_depth, depth = depth + 1)
            self.right = GPTree(Parameters.gp_rules)
            self.right.random_tree(grow, max_depth, depth = depth + 1)
        elif self.data in Parameters.gp_rules['unary_functions']:
            self.left = GPTree(Parameters.gp_rules)
            self.left.random_tree(grow, max_depth, depth = depth + 1)

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
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree

class GeneticProgram():
    """Class representing a genetic program.

    Attributes
    ----------
    population : (n,) np.array, dtype=GPTree
        current population of solutions
    fitnesses : (n,) np.array
        array of fitnesses of all solutions

    """

    def __init__(self):
        self.population = np.empty((Parameters.gp_rules['pop_size'],), dtype=GPTree)
        self.fitnesses = np.zeros((Parameters.gp_rules['pop_size']))

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

    # TODO: in the future, here we will add another parameter - indexes in dataset, which to use - our fitness predictor
    def fitness(self, individual, dataset):
        """fitness function of the program

        Parameters
        ----------
        individual : GPTree
            individual we want to evaluate
        dataset : (m, d) np.array
            set of m test, values at 0..d-2 are inputs, value at d-1 is the expected output

        Returns
        -------
        float
            mean absolute error on all tests

        """
        return mean([abs(individual.compute_tree(ds[:-1]) - ds[-1]) for ds in dataset]) + 0.01*individual.size()

    def selection(self):
        """Selects an andividual using tournament selection.

        Returns
        -------
        GPTree
            copy of the winner of the tournament

        """
        tournament = [randint(0, len(self.population) - 1) for _ in range(Parameters.gp_rules['tournament_size'])] # select tournament contenders
        tournament_fitnesses = [self.fitnesses[contender] for contender in tournament]
        return deepcopy(self.population[tournament[np.argmin(tournament_fitnesses))]])

    def evaluate_population(self, dataset):
        """computes and saves fitness of the population

        Parameters
        ----------
        dataset : (m, d) np.array
            see function fitness for explanation

        """
        for i in range(len(self.fitnesses)):
            self.fitnesses[i] = self.fitness(self.population[i], dataset)

    def evolve_population(self):
        """Evolves the current population using mutation and crossover

        """
        for i in range(Parameters.gp_rules['pop_size']):
            parent1 = self.selection()
            parent2 = self.selection()
            parent1.crossover(parent2)
            parent1.mutation()
            self.population[i] = parent1

    def run_evolution(self, dataset, verbose=False):
        """perform an evolution run

        Parameters
        ----------
        dataset : (m, d) np.array
            dataset to use
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

        # start
        cur_gen = 1
        while not end_criteria_met:
            cur_gen += 1
            if verbose and cur_gen % 100 == 0: print(f'current generation {cur_gen}')
            self.evolve_population()
            #self.init_population()
            self.evaluate_population(dataset)

            best_of_gen = np.argmin(self.fitnesses)
            avg_fitnesses.append(mean(self.fitnesses))
            best_of_gen_fitnesses.append(self.fitnesses[best_of_gen])
            avg_sizes.append(mean([t.size() for t in self.population]))

            if self.fitnesses[best_of_gen] < best_of_run_f:
                best_of_run_f = self.fitnesses[best_of_gen]
                best_of_run_gen = cur_gen
                best_of_run = deepcopy(self.population[best_of_gen])

            best_of_run_fitnesses.append(best_of_run_f)

            end_criteria_met |= cur_gen > Parameters.gp_rules['generations'] or best_of_run_f < 0.1
        return {'best' : best_of_run,
                'generations' : cur_gen,
                'avg_fitnesses' : avg_fitnesses,
                'best_of_gen_fitnesses' : best_of_gen_fitnesses,
                'best_of_run_fitnesses' : best_of_run_fitnesses,
                'best_f' : best_of_run_f,
                'avg_sizes' : avg_sizes}

if __name__== "__main__":
    pass
