#! /usr/bin/env python

"""
sea_bin.py
A very simple EA for binary representation.
Ernesto Costa Março 2015
"""

__author__ = 'Ernesto Costa'
__date__ = 'March 2015'

from random import random, randint, sample, shuffle
from operator import itemgetter
import matplotlib.pyplot as plt


def sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
        sel_survivors, fitness_func):

    accumulated_generations = []
    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)
    accumulated_generations.append(populacao)
    # avalia população
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

    for i in range(numb_generations):
        # selecciona progenitores
        mate_pool = sel_parents(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for j in range(0, size_pop - 1, 2):
            cromo_1 = mate_pool[j]
            cromo_2 = mate_pool[j + 1]
            filhos = recombination(cromo_1, cromo_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation
        descendentes = []
        for indiv, fit in progenitores:
            novo_indiv = mutation(indiv, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        # store the population
        accumulated_generations.append(populacao)
    return accumulated_generations


def sea_plot(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
             sel_survivors, fitness_func):
    all_best = []
    all_average = []

    for i in range(3):

        print(i)

        # inicializa população: indiv = (cromo,fit)
        populacao = gera_pop(size_pop, size_cromo)
        # avalia população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

        for j in range(numb_generations):
            # selecciona progenitores
            mate_pool = sel_parents(populacao)
            # Variation
            # ------ Crossover
            progenitores = []
            for k in range(0, size_pop - 1, 2):
                cromo_1 = mate_pool[k]
                cromo_2 = mate_pool[k + 1]
                filhos = recombination(cromo_1, cromo_2, prob_cross)
                progenitores.extend(filhos)
                # ------ Mutation
            descendentes = []
            for indiv, fit in progenitores:
                novo_indiv = mutation(indiv, prob_mut)
                descendentes.append((novo_indiv, fitness_func(novo_indiv)))
            # New population
            populacao = sel_survivors(populacao, descendentes)
            # Avalia nova _população
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

        all_best.append(best_pop(populacao)[1])
        all_average.append(average_pop(populacao))

    plt.figure()
    plt.plot(all_best)
    plt.plot(all_average)
    plt.show()


# Representation by binary strings

def gera_pop(size_pop, size_cromo):
    return [(gera_indiv(size_cromo), 0) for i in range(size_pop)]


def gera_indiv(size_cromo):
    indiv = [randint(0, 1) for i in range(size_cromo)]
    return indiv


# Binary mutation

def muta_bin(indiv, prob_muta):
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_bin_gene(cromo[i], prob_muta)
    return cromo


def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g


# Crossover
def one_point_cross(cromo_1, cromo_2, prob_cross):
    value = random()
    if value < prob_cross:
        pos = randint(0, len(cromo_1))
        f1 = cromo_1[0:pos] + cromo_2[pos:]
        f2 = cromo_2[0:pos] + cromo_1[pos:]
        return (f1, f2)
    else:
        return (cromo_1, cromo_2)


def two_points_cross(cromo_1, cromo_2, prob_cross):
    value = random()
    if value < prob_cross:
        pc = sample(range(len(cromo_1)), 2)
        pc.sort()
        pc1, pc2 = pc
        f1 = cromo_1[:pc1] + cromo_2[pc1:pc2] + cromo_1[pc2:]
        f2 = cromo_2[:pc1] + cromo_1[pc1:pc2] + cromo_2[pc2:]
        return (f1, f2)
    else:
        return (cromo_1, cromo_2)


def uniform_cross(cromo_1, cromo_2, prob_cross):
    value = random()
    if value < prob_cross:
        f1 = []
        f2 = []
        for i in range(0, len(cromo_1)):
            if random() < 0.5:
                f1.append(cromo_1[i])
                f2.append(cromo_2[i])
            else:
                f1.append(cromo_2[i])
                f2.append(cromo_1[i])

        return (f1, f2)
    else:
        return (cromo_1, cromo_2)


# Tournament Selection
def tour_sel(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = tour(pop, t_size)
            mate_pool.append(winner)
        return mate_pool

    return tournament


def tour(population, size):
    """Maximization Problem.Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=True)
    return pool[0]


# Survivals: elitism
def sel_survivors_elite(elite):
    def elitism(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism


# Fitness
def merito(indiv):
    return evaluate(fenotipo(indiv))


# auxiliary
def display(indiv, phenotype, problem):
    print('Chromo: %s\nFitness: %s' % (phenotype(indiv[0], problem), indiv[1]))


def best_pop(populacao):
    populacao.sort(key=itemgetter(1), reverse=True)
    return populacao[0]


def average_pop(populacao):
    return sum([fit for crom, fit in populacao]) / len(populacao)


# -------------------  Problem Specific Definitions
# One max

def fenotipo(indiv):
    return indiv


def evaluate(indiv):
    return sum(indiv)
