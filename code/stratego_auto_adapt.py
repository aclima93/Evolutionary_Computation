"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: This is mostly a wrapper, but it's ours
"""

from random import *
from kp_1 import *


def average_pop(refference_window, fitness_func):

    # to get an average population get the average ith individual in each population fot the refference window

    num_populations = len(refference_window)

    if num_populations > 1:
        average_population = []
        ith_pairings = list(zip(*refference_window))
        for ith_pair in ith_pairings:
            average_population.append(average_indiv(ith_pair, fitness_func))
        return average_population

    else:
        return refference_window[0]

def average_indiv(population, fitness_func):

    individuals = []
    for indiv, fit in population:
        individuals.append(indiv)

    num_individuals = len(individuals)
    average_individual = [0] * len(individuals[0])  # individual with only zeros
    # sum all the solutions
    for indiv in individuals:
        for i in range(len(indiv)):
            average_individual[i] += indiv[i]

    # divide by number of individuals in population
    for i in range(len(average_individual)):
        average_individual[i] = int( average_individual[i] / num_individuals )

    return [average_individual, fitness_func(average_individual)]

"""
compares individuals by differences in genotype
"""
def compare_individs(temp1, temp2):
    s = set(temp2[0])
    temp3 = [x for x in temp1[0] if x not in s]
    return temp3

"""
tendo em conta a janela de referência, decidir se devem ser alterados os parâmetros:
- tamanho da população (?)
- probabilidade de crossover
- probabilidade de mutação
"""
def auto_adapt_fitness(cur_population, refference_window, fitness_func):

    print("\n\n----------------------------------")
    print("cur_population: ")
    print(cur_population)
    print("refference_window: ")
    print(refference_window)

    # get average population from refference_window
    average_reff_population = average_pop(refference_window, fitness_func)

    print("average_reff_population: ")
    print(average_reff_population)

    # get average individual from average population
    average_reff_individual = average_indiv(average_reff_population, fitness_func)

    print("average_reff_individual: ")
    print(average_reff_individual)

    # get average individual from current population
    average_individual = average_indiv(cur_population, fitness_func)

    print("average_individual: ")
    print(average_individual)

    # compare average individuals
    differences = compare_individs(average_reff_individual, average_individual)

    print("num differences: ")
    print(len(differences))

    # return ratio between number of differences and size of current average_individual
    return len(differences) / len(average_individual)



"""
Survivals: elitism
tendo em conta a janela de referência, alterar o tamanho da população
"""
def stratego_next_population(elite):

    def elitism(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]

        return new_population

    return elitism


"""
funcao stratego
- variação da sea fornecida pelo prof. Ernesto mas que aplica os conceitos que estamos a estudar:
--- variação do tamanho da população, prob. de mutação e prob. de crossover
"""
def stratego(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
             sel_survivors, fitness_func, refference_window_size, refference_window):

    # TODO: cross & mut prob alteration based on refference window

    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)

    # avalia população
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

    for i in range(numb_generations):

        # selecciona progenitores
        mate_pool = sel_parents(populacao)

        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, size_pop - 1, 2):
            cromo_1 = mate_pool[i]
            cromo_2 = mate_pool[i + 1]
            filhos = recombination(cromo_1, cromo_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation

        descendentes = []

        for indiv, fit in progenitores:
            novo_indiv = mutation(indiv, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))

        # added previous generation to refference_window and remove the oldest (or just append until we have enough)
        if len(refference_window) == refference_window_size:
            refference_window.remove(refference_window[0])
        refference_window.append(populacao)

        # New population
        populacao = sel_survivors(populacao, descendentes)

        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

        # add or remove individuals to new population based on refference window
        ratio = auto_adapt_fitness(populacao, refference_window, fitness_func)

        if ratio < 0.25:  # start reversing the crossover and mutation

            if (prob_cross - ratio) > 0.0:
                prob_cross -= ratio

            if (prob_mut + ratio) < 1.0:
                prob_mut += ratio


    return best_pop(populacao)


"""
função run
- executa uma simulação com os parâmetros fornecidos
- devolve os resultados da experiência
"""
def run(auto_adapt, problem, size_items):
    refference_window_size = 3  # number previous generations considered for altering the auto-adaptative parameters
    refference_window = []

    num_generations = 10 # 500
    population_size = 5 # 250
    prob_crossover = 0.80  # resposável por salto grande no início
    prob_mutation = 0.10  # resposável por saltos pequenos no final

    if not auto_adapt:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        # mutation, sel_survivors, fitness_func)
        best = sea(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                   one_point_cross, muta_bin, sel_survivors_elite(0.02), merito(problem))
    else:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        # mutation, sel_survivors, fitness_func, refference_window_size, refference_window)
        best = stratego(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                        one_point_cross, muta_bin, stratego_next_population(0.02), merito(problem),
                        refference_window_size, refference_window)

    return [best, phenotype, problem]


"""
funcao run_n_times
- tem todos os parâmetros que devem ser analisados estatisticamente
--- os parâmetros, segundo o enunciado, terão a ver com um desvio padrão e uma média utilizados
--- aquando da mutação e recombinação
- executa a função run n vezes
- guarda os resultados e parâmetros de execução num ficheiro
"""
def run_n_times(num_runs):

    size_items = 10
    max_value = 10

    results_with_auto_adapt = []
    results_without_auto_adapt = []

    for i in range(num_runs):
        problem = generate_uncor(size_items, max_value)

        results_with_auto_adapt.append(run(True, problem, size_items))
        results_without_auto_adapt.append(run(False, problem, size_items))

    return [results_with_auto_adapt, results_without_auto_adapt]


"""
função de análise estatística e apresentação de gráficos
- analisar os melhores resultados e os resultados da média
- analisar o efeito das alterações nos parâmetros
"""
# TODO: finish it!
def analyse(data):
    for [best, phenotype, problem] in data:
        display(best, phenotype, problem)

    return


"""
starting point for our algorithm
"""
if __name__ == '__main__':

    seed(666)  # random number generation with fixed seed for reproduceable results

    number_of_runs = 1  # TODO: 30  # statistically relevant ammount of runs

    results = run_n_times(number_of_runs)

    print("\n\n----- Analysing results without Auto-Adapt -----")
    analyse(results[0])

    print("\n\n----- Analysing results with Auto-Adapt -----")
    analyse(results[1])
