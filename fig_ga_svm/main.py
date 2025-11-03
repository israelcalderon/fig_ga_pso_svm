import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


import genetic_algorithm as ga


FIXED_BANDS = [] # fixed genes that represent RGB


# def svm(X, y) -> float:
#     svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=42))
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#     f1_scores = cross_val_score(svm_clf, X, y, cv=cv, scoring='f1')
#     return np.mean(f1_scores)


def svm(X, y) -> float:
    pipeline = make_pipeline(StandardScaler(), SVC(random_state=42, class_weight='balanced'))
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 0.01, 0.1, 1],
        'svc__kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_score_


def objective_fun(individual: tuple[str], X: pd.DataFrame, y: pd.DataFrame) -> float:
    columns = FIXED_BANDS + list(individual)
    x = X[columns]
    return svm(x, y)


def main():
    df = pd.read_csv('/Users/israel/Projects/figs_means_db/db/figs_means.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.2)
    population = ga.create_population()
    best_global_individual = None
    best_global_fitness = -1
    best_individuals_fitness_by_generation = []
    stagnation_count = 0
    best_fitness_improved = False
    initial_mutation_rate = ga.MUTATION_RATE

    print("--- Iniciando Algoritmo Genético ---")

    for generation in range(ga.NUM_GENERATIONS):
        population_fitness = [(ind, objective_fun(ind, X_train, y_train)) for ind in population]

        # Ordenamos la población por fitness (de mayor a menor)
        population_fitness.sort(key=lambda item: item[1], reverse=True)
        pprint.pp(population_fitness)
        # Actualizamos el mejor individuo encontrado hasta ahora
        best_generation_individual = population_fitness[0]
        if best_generation_individual[1] > best_global_fitness:
            best_global_individual = best_generation_individual[0]
            best_global_fitness = best_generation_individual[1]
        
        print(f"Generación {generation+1}/{ga.NUM_GENERATIONS} | Mejor Fitness: {best_generation_individual[1]:.2f} | Mejor Individuo: {best_generation_individual[0]}")
        best = (best_generation_individual[1], best_generation_individual[0])
        if generation > 0 and best == best_individuals_fitness_by_generation[-1]:
            stagnation_count += 1
            best_fitness_improved = False
        
        if generation > 0 and best > best_individuals_fitness_by_generation[-1]:
            best_fitness_improved = True

        best_individuals_fitness_by_generation.append(best)
        # 4. Creamos la siguiente generación
        siguiente_generacion = []

        # 4.1. Elitismo: Los mejores pasan directamente
        for j in range(ga.ELITISM_COUNT):
            siguiente_generacion.append(population_fitness[j][0])

        # 4.2. Creamos el resto de la nueva generación
        while len(siguiente_generacion) < ga.POPULATION_SIZE:
            padre1 = ga.tournament_selection(population_fitness)
            padre2 = ga.tournament_selection(population_fitness)
            
            # Cruce
            hijo1, hijo2 = ga.cruce(padre1, padre2)
            
            # Mutación
            hijo1 = ga.mutacion(hijo1)
            hijo2 = ga.mutacion(hijo2)
            
            siguiente_generacion.append(hijo1)
            if len(siguiente_generacion) < ga.POPULATION_SIZE:
                siguiente_generacion.append(hijo2)

        if stagnation_count >= ga.STAGNATION_THRESHOLD:
            print("Estancamiento alcanzado, incrementando probabilidad de mutación...")
            ga.MUTATION_RATE = ga.STAGNATION_MUTATION_RATE
            stagnation_count = 0
        
        if ga.MUTATION_RATE == ga.STAGNATION_MUTATION_RATE and best_fitness_improved:
            print("Fitness mejorado, restaurando probabilidad de mutación...")
            ga.MUTATION_RATE = initial_mutation_rate
            stagnation_count = 0

        # 5. Replace new generation
        population = siguiente_generacion

    print("\n--- Algoritmo Finalizado ---")
    print(f"Mejor individuo encontrado: {best_global_individual}")
    print(f"Valor de fitness (F1 Score): {best_global_fitness:.4f}")
    
    print("Mejores individuos por generación:")
    print(best_individuals_fitness_by_generation)


if __name__ == '__main__':
    main()
