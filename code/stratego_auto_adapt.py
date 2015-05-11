"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: This is mostly a wrapper, but it's ours
"""

from random import *
from kp_1 import *


"""
tendo em conta a janela de referência, decidir se devem ser alterados os parâmetros:
- tamanho da população
- probabilidade de crossover
- probabilidade de mutação
"""
# TODO: finish it!
def auto_adapt_fitness():
    return


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

        # TODO: add or remove individuals to new population based on refference window

        return new_population

    return elitism


"""
funcao stratego
- variação da sea fornecida pelo prof. Ernesto mas que aplica os conceitos que estamos a estudar:
--- variação do tamanho da população, prob. de mutação e prob. de crossover
"""
def stratego(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
             sel_survivors, fitness_func, refference_window):

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

        # New population
        populacao = sel_survivors(populacao, descendentes)

        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

    return best_pop(populacao)


"""
função run
- executa uma simulação com os parâmetros fornecidos
- devolve os resultados da experiência
"""
def run(auto_adapt, problem, size_items):

    refference_window = 1  # number previous generations to eb considered for altering the auto-adaptative parameters

    num_generations = 500
    population_size = 250  # the number of elements influences the effect fo reproduction because the gene pool is bigger
    prob_crossover = 0.80  # resposável por salto grande no início
    prob_mutation = 0.10  # resposável por saltos pequenos no final

    if not auto_adapt:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        #     mutation, sel_survivors, fitness_func)
        best = sea(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                   one_point_cross, muta_bin, sel_survivors_elite(0.02), merito(problem))
    else:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        #     mutation, sel_survivors, fitness_func, refference_window)
        best = stratego(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                        one_point_cross, muta_bin, stratego_next_population(0.02), merito(problem), refference_window)

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

        results_with_auto_adapt.append( run( True, problem, size_items))
        results_without_auto_adapt.append( run( False, problem, size_items))

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
