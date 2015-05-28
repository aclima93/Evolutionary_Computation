from sea_bin import *
import random
import copy
import operator

# 0/1 Knapsack Problem


# Evaluation
def merito(problem):
    """
    problem is a dictionary describing the instance of the KP.
    keys = values (list of invidual's values), weigths (list of individual's weights),capacity (a float with the total capacity)
    """

    def fitness(indiv):
        quali = evaluate_zero(phenotype(indiv, problem), problem)
        return quali

    return fitness


def phenotype(indiv, problem):
    """from a binary string to a list of [id,weight,value]."""
    pheno = [[id_, problem['weights'][id_], problem['values'][id_]] for id_ in range(len(indiv)) if indiv[id_] == 1]
    return pheno


def evaluate_zero(feno, problem):
    """ feno = [...,[ıd,weight,value],...]"""
    total_weight = sum([weight for id_, weight, value in feno])
    if total_weight > problem['capacity']:
        return 0
    return sum([value for id_, weight, value in feno])


# Data Sets
def generate_uncor(size_items, max_value):
    weights = [random.uniform(1, max_value) for _ in range(size_items)]
    values = [random.uniform(1, max_value) for _ in range(size_items)]
    capacity = int(0.5 * sum(weights))
    return {'weights': weights, 'values': values, 'capacity': capacity}


def generate_weak_cor(size_items, max_value, amplitude):
    weights = [random.uniform(1, max_value) for _ in range(size_items)]
    values = []
    for i in range(size_items):
        value = weights[i] + random.uniform(-amplitude, amplitude)
        while value <= 0:
            value = weights[i] + random.uniform(-amplitude, amplitude)
        values.append(value)
    capacity = int(0.5 * sum(weights))
    return {'weights': weights, 'values': values, 'capacity': capacity}


def generate_strong_cor(size_items, max_value, amplitude):
    weights = [random.uniform(1, max_value) for _ in range(size_items)]
    values = [weights[i] + amplitude for i in range(size_items)]
    capacity = int(0.5 * sum(weights))
    return {'weights': weights, 'values': values, 'capacity': capacity}


# Repair individuals
def repair_weight(cromo, problem):
    """repair an individual be eliminating items using the least weighted gene."""
    indiv = copy.deepcopy(cromo)
    capacity = problem['capacity']
    pheno = phenotype(indiv, problem)
    pheno.sort(key=operator.itemgetter(1))

    weight_indiv = get_weight(indiv, problem)
    for index, weight, value in pheno:
        if weight_indiv <= capacity:
            return indiv
        else:
            indiv[index] = 0
            weight_indiv -= weight


def repair_value(cromo, problem):
    """repair an individual be eliminating items using the least valued gene."""
    indiv = copy.deepcopy(cromo)
    capacity = problem['capacity']
    pheno = phenotype(indiv, problem)
    pheno.sort(key=operator.itemgetter(2))

    weight_indiv = get_weight(indiv, problem)
    for index, weight, value in pheno:
        if weight_indiv <= capacity:
            return indiv
        else:
            indiv[index] = 0
            weight_indiv -= weight


def repair_value_to_profit(cromo, problem):
    """repair an individual be eliminating items using the ratio value/weight."""
    indiv = copy.deepcopy(cromo)
    capacity = problem['capacity']
    pheno = phenotype(indiv, problem)
    pheno = [[i, w, v, float(v / w)] for i, w, v in pheno]
    pheno.sort(key=operator.itemgetter(3))

    weight_indiv = get_weight(indiv, problem)
    for index, weight, value, ratio in pheno:
        if weight_indiv <= capacity:
            return indiv
        else:
            indiv[index] = 0
            weight_indiv -= weight


def get_weight(indiv, problem):
    total_weight = sum([problem['weights'][gene] for gene in range(len(indiv)) if indiv[gene] == 1])
    return total_weight


def get_value(indiv, problem):
    total_value = sum([problem['values'][gene] for gene in range(len(indiv)) if indiv[gene] == 1])
    return total_value


'''
if __name__ == '__main__':
    problem = generate_strong_cor(10, 10, 5)
    fit = merito(problem)
    best = sea(100, 50, 10, 0.1, 0.8, tour_sel(3), one_point_cross, muta_bin, sel_survivors_elite(0.02), fit)
    display(best, phenotype)
'''




