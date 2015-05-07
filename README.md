# Evolutionary_Computation

# 3.6 TP6 - Auto-Adaptação em Estratégias Evolutivas

** Descrição Sumária **

Fazer variar os parâmetros de uma experiência ao longo da simulação parece dar bons resultados. Por exemplo, o tamanho da população ou as probabilidades dos operadores de variação. Podemos incluir na nossa implementação o modo como isso é feito. Mas melhor seria deixar o processo evolutivo tratar desse assunto. É isso que vamos estudar, no contexto das estratégias evolutivas, variante dos algoritmos evolucionários.

** Objectivos **

Como sabe, as EE permitem incluir na representação das soluções os parâmetros estratégicos, como os valores dos desvios padrão da distribuição normal que usamos para efectuar a mutação de cada gene. Escolha uma implementação em Python das estratégias evolutivas, do tipo (μ + λ). Use como problema de referência para fazer o seu estudo, um dos problemas de optimização de funções. Ver descrição dos problemas possíveis na secção ??. Use duas versões do algoritmo, uma, sem e outra com auto-adaptação.

Para cada algoritmo faça 30 testes (runs). Recolha os dados sobre desempenho de cada algoritmo, medido pela qualidade do resultado. Analise estatisticamente os resultados e tire conclusões.