"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: This is mostly a wrapper, but it's ours
"""

from random import *
from kp_1 import *


# tendo em conta a janela de referência, decidir se devem ser alterados os parâmetros
# - tamanho da população
# - probabilidade de crossover
# - probabilidade de mutação
def auto_adapt_fitness():
    return


def stratego(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
             sel_survivors, fitness_func, refference_window):

    # TODO: apply population size alteration + cross & mut prob alteration based on refference window

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
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

    return best_pop(populacao)


# criar a função run
# - executa uma simulação com os parâmetros fornecidos
# - devolve os resultados da experiência
def run(auto_adapt, refference_window, problem, size_items, num_generations, population_size, prob_crossover,
        prob_mutation):
    if not auto_adapt:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        #     mutation, sel_survivors, fitness_func)
        best = sea(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                   one_point_cross, muta_bin, sel_survivors_elite(0.02), merito(problem))
    else:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        #     mutation, sel_survivors, fitness_func, refference_window)
        best = stratego(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                        one_point_cross, muta_bin, sel_survivors_elite(0.02), merito(problem), refference_window)

    return [best, phenotype, problem]


# criar a função run_n_times
# - tem todos os parâmetros que devem ser analisados estatisticamente
# --- os parâmetros, segundo o enunciado, terão a ver com um desvio padrão e uma média utilizados
# --- aquando da mutação e recombinação
# - executa a função run n vezes
# - guarda os resultados e parâmetros de execução num ficheiro
def run_n_times(num_runs, refference_window, num_generations, population_size, prob_crossover, prob_mutation):
    size_items = 10
    max_value = 10

    results_with_auto_adapt = []
    results_without_auto_adapt = []

    for i in range(num_runs):
        problem = generate_uncor(size_items, max_value)

        results_with_auto_adapt.append(
            run(True, refference_window, problem, size_items, num_generations, population_size, prob_crossover,
                prob_mutation))
        results_without_auto_adapt.append(
            run(False, refference_window, problem, size_items, num_generations, population_size, prob_crossover,
                prob_mutation))

    return results_with_auto_adapt, results_without_auto_adapt


# criar a função de análise estatística e apresentação de gráficos
# - analisar os melhores resultados e os resultados da média
# - analisar o efeito das alterações nos parâmetros
def analyse(data):
    for [best, phenotype, problem] in data:
        display(best, phenotype, problem)

    return


# starting point for our algorithm
if __name__ == '__main__':
    seed(666)  # random number generation with fixed seed for reproduceable results

    number_of_runs = 1  # TODO: 30  # statistically relevant ammount of runs

    refference_window = 1  # number previous generations to eb considered for altering the auto-adaptative parameters

    num_generations = 500
    population_size = 250  # the number of elements influences the effect fo reproduction because the gene pool is bigger
    prob_crossover = 0.80  # resposável por salto grande no início
    prob_mutation = 0.10  # resposável por saltos pequenos no final

    results_with_auto_adapt, results_without_auto_adapt = run_n_times(number_of_runs, refference_window,
                                                                      num_generations, population_size, prob_crossover,
                                                                      prob_mutation)

    print("\n\n----- Analysing results without Auto-Adapt -----")
    analyse(results_without_auto_adapt)

    print("\n\n----- Analysing results with Auto-Adapt -----")
    analyse(results_with_auto_adapt)
