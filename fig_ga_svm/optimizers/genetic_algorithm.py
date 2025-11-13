import random
from typing import List, Tuple
from joblib import Parallel, delayed

from fig_ga_svm.evaluator import FitnessEvaluator
from fig_ga_svm.optimizers.optimizer import Optimizer, Individual, Fitness, History
from fig_ga_svm.data import GENE_POOL


class GeneticAlgorithmOptimizer(Optimizer):

    def _create_individual(self, num_genes: int) -> Individual:
        if num_genes > len(GENE_POOL):
            raise ValueError("num_genes cannot be larger than GENE_POOL")
        return tuple(random.sample(GENE_POOL, num_genes))

    def _create_population(self, pop_size: int, num_genes: int) -> List[Individual]:
        population: List[Individual] = []
        while len(population) < pop_size:
            ind = self._create_individual(num_genes)
            if ind not in population:
                population.append(ind)
        return population

    def _tournament_selection(self, population_fitness: List[Tuple[Individual, Fitness]], tournament_size: int) -> Individual:
        competitors = random.sample(population_fitness, min(tournament_size, len(population_fitness)))
        winner = max(competitors, key=lambda item: item[1])
        return winner[0]

    def _roulette_wheel_selection(self, population_fitness: List[Tuple[Individual, Fitness]]) -> Individual:
        fitnesses = [f for _, f in population_fitness]
        total = sum(fitnesses)
        if total <= 0:
            # fallback to best individual if all fitnesses are zero
            return max(population_fitness, key=lambda x: x[1])[0]
        pick = random.random() * total
        current = 0.0
        for ind, fit in population_fitness:
            current += fit
            if current >= pick:
                return ind
        return population_fitness[-1][0]

    def _single_point_crossover(self, parent1: Individual, parent2: Individual, num_genes: int) -> Tuple[Individual, Individual]:
        # choose cut in [1, num_genes-1]
        if num_genes <= 1:
            return parent1, parent2
        cut = random.randint(1, num_genes - 1)
        child1 = list(parent1[:cut] + parent2[cut:])
        child2 = list(parent2[:cut] + parent1[cut:])

        def repair(child: List[str]) -> Tuple[str, ...]:
            # ensure uniqueness within child by replacing duplicates
            used = set()
            available = [g for g in GENE_POOL if g not in child]
            repaired: List[str] = []
            for gene in child:
                if gene in used:
                    if not available:
                        # no available replacements (shouldn't happen unless pool smaller than num_genes)
                        raise ValueError("Not enough unique genes to repair child")
                    replacement = available.pop(random.randrange(len(available)))
                    repaired.append(replacement)
                    used.add(replacement)
                else:
                    repaired.append(gene)
                    used.add(gene)
            # if repaired shorter or longer adjust (shouldn't happen), truncate/pad
            return tuple(repaired[:num_genes])

        return repair(child1), repair(child2)

    def _mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        indiv_list = list(individual)
        for i, gene in enumerate(indiv_list):
            if random.random() < mutation_rate:
                candidates = [g for g in GENE_POOL if g not in indiv_list]
                if not candidates:
                    continue
                indiv_list[i] = random.choice(candidates)
        return tuple(indiv_list)

    def optimize(
        self,
        evaluator: FitnessEvaluator,
        meta_heuristic_params: dict[str, int | float]
    ) -> tuple[Individual, Fitness, History]:
        num_genes = meta_heuristic_params['num_bands']
        population_size = meta_heuristic_params['population_size']
        num_generations = meta_heuristic_params['num_iterations']
        elitism_count = meta_heuristic_params['elitism_count']
        mutation_rate = meta_heuristic_params['mutation_rate']
        stagnation_mutation_rate = meta_heuristic_params['stagnation_mutation_rate']
        stagnation_threshold = meta_heuristic_params['stagnation_threshold']
        tournament_size = meta_heuristic_params['tournament_size']
        selection = meta_heuristic_params['selection_type']

        if num_genes > len(GENE_POOL):
            raise ValueError('num_genes cannot be larger than GENE_POOL size')

        population = self._create_population(population_size, num_genes)

        best_global_individual: Individual = None  # type: ignore[assignment]
        best_global_fitness: float = float('-inf')
        history: List[Tuple[Individual, Fitness]] = []

        stagnation_count = 0
        best_prev: Tuple[Fitness, Individual] | None = None
        initial_mutation_rate = mutation_rate

        print("\n--- 🧬 Iniciando Optimización con Algoritmo Genetico ---")

        for generation in range(num_generations):
            population_fitness = Parallel(n_jobs=-1)(
                delayed(lambda individual: (individual, float(evaluator.evaluate(individual))))(individual) for individual in population
            )

            # Sort descending by fitness
            population_fitness.sort(key=lambda x: x[1], reverse=True)

            best_gen_ind, best_gen_fit = population_fitness[0]
            if best_gen_fit > best_global_fitness:
                best_global_fitness = best_gen_fit
                best_global_individual = best_gen_ind

            print(f"Generation {generation+1}/{num_generations} | Best fitness: {best_gen_fit:.4f} | Best ind: {best_gen_ind}")
            history.append((best_gen_ind, best_gen_fit))

            current_best = (best_gen_fit, best_gen_ind)
            if best_prev is not None:
                if current_best == best_prev:
                    stagnation_count += 1
                elif current_best[0] > best_prev[0]:
                    # improved
                    stagnation_count = 0
            best_prev = current_best

            # adapt mutation on stagnation
            if stagnation_count >= stagnation_threshold:
                print("Stagnation reached, increasing mutation rate...")
                mutation_rate = stagnation_mutation_rate
                stagnation_count = 0
            elif mutation_rate == stagnation_mutation_rate and best_prev and best_prev[0] > history[-1][1]:
                # restore if improved
                mutation_rate = initial_mutation_rate

            # Create next generation
            next_generation: List[Individual] = []

            # Elitism
            for i in range(min(elitism_count, len(population_fitness))):
                next_generation.append(population_fitness[i][0])

            # Fill rest
            while len(next_generation) < population_size:
                if selection == 'roulette':
                    parent1 = self._roulette_wheel_selection(population_fitness)
                    parent2 = self._roulette_wheel_selection(population_fitness)
                else:
                    parent1 = self._tournament_selection(population_fitness, tournament_size)
                    parent2 = self._tournament_selection(population_fitness, tournament_size)

                child1, child2 = self._single_point_crossover(parent1, parent2, num_genes)
                child1 = self._mutate(child1, mutation_rate)
                child2 = self._mutate(child2, mutation_rate)

                if child1 not in next_generation:
                    next_generation.append(child1)
                if len(next_generation) < population_size and child2 not in next_generation:
                    next_generation.append(child2)

            population = next_generation
        return best_global_individual, best_global_fitness, history
