import math
import random
from typing import List, Tuple, Any
from joblib import Parallel, delayed

from fig_ga_svm.evaluator import FitnessEvaluator
from fig_ga_svm.optimizers.optimizer import Optimizer, Individual, Fitness, History
from fig_ga_svm.data import GENE_POOL

# Individual is now a tuple of bits (0/1)
BitVector = Tuple[int, ...]

class GeneticAlgorithmOptimizer(Optimizer):

    def _create_individual(self, num_genes: int) -> BitVector:
        bits = [0] * len(GENE_POOL)
        ones_indices = random.sample(range(len(GENE_POOL)), num_genes)
        for idx in ones_indices:
            bits[idx] = 1
        return tuple(bits)

    def _create_population(self, pop_size: int, num_genes: int) -> List[BitVector]:
        population: List[BitVector] = []
        while len(population) < pop_size:
            ind = self._create_individual(num_genes)
            if ind not in population:
                population.append(ind)
        return population

    def _bitvector_to_genes(self, bitvector: BitVector) -> Tuple[str, ...]:
        return tuple([GENE_POOL[i] for i, bit in enumerate(bitvector) if bit == 1])

    def _tournament_selection(self, population_fitness: List[Tuple[BitVector, Fitness]], tournament_size: int) -> BitVector:
        competitors = random.sample(population_fitness, min(tournament_size, len(population_fitness)))
        winner = max(competitors, key=lambda item: item[1])
        return winner[0]

    def _roulette_wheel_selection(self, population_fitness: List[Tuple[BitVector, Fitness]]) -> BitVector:
        fitnesses = [f for _, f in population_fitness]
        total = sum(fitnesses)
        if total <= 0:
            return max(population_fitness, key=lambda x: x[1])[0]
        pick = random.random() * total
        current = 0.0
        for ind, fit in population_fitness:
            current += fit
            if current >= pick:
                return ind
        return population_fitness[-1][0]

    def _single_point_crossover(self, parent1: BitVector, parent2: BitVector, num_genes: int) -> Tuple[BitVector, BitVector]:
        length = len(GENE_POOL)
        if length <= 1:
            return parent1, parent2
        cut = random.randint(1, length - 1)
        child1 = list(parent1[:cut] + parent2[cut:])
        child2 = list(parent2[:cut] + parent1[cut:])

        def repair(child: List[int]) -> BitVector:
            # Ensure exactly num_genes bits are set to 1
            ones = [i for i, b in enumerate(child) if b == 1]
            zeros = [i for i, b in enumerate(child) if b == 0]
            if len(ones) > num_genes:
                # Randomly set excess ones to zero
                for idx in random.sample(ones, len(ones) - num_genes):
                    child[idx] = 0
            elif len(ones) < num_genes:
                # Randomly set zeros to one
                for idx in random.sample(zeros, num_genes - len(ones)):
                    child[idx] = 1
            return tuple(child)
        return repair(child1), repair(child2)

    def __blend_parents(self, parent1: BitVector, parent2: BitVector) -> BitVector:
        return tuple([parent1[bit] or parent2[bit] for bit in range(len(parent1))])

    def __find_indexes(self, vector: BitVector, val: Any) -> list[int]:
        return [idx for idx, v in enumerate(vector) if v == val]

    def _blx_crossover(self,
                       parent1: BitVector,
                       parent2: BitVector,
                       num_genes: int,
                       alpha: float = 0.0) -> tuple[BitVector, BitVector]:
        # BLX-⍺ is for real-valued genes; for binary, we blend parent bits and expand selection window
        min_limit, max_limit = 0, len(GENE_POOL) - 1
        merged_parents = self.__blend_parents(parent1, parent2)
        bit_positions = self.__find_indexes(merged_parents, 1)
        if not bit_positions:
            # fallback: random selection if no bits are set
            bit_positions = list(range(len(GENE_POOL)))
        distance = max(bit_positions) - min(bit_positions)
        min_expanded = max(min_limit, min(bit_positions) - math.ceil(distance * alpha))
        max_expanded = min(max_limit, max(bit_positions) + math.ceil(distance * alpha))

        # Select unique indices for each child
        possible_indices = list(range(min_expanded, max_expanded + 1))
        if len(possible_indices) < num_genes:
            # fallback: use all possible indices
            possible_indices = list(range(len(GENE_POOL)))

        child1_indices = random.sample(possible_indices, num_genes)
        child2_indices = random.sample(possible_indices, num_genes)

        child1 = [1 if i in child1_indices else 0 for i in range(len(GENE_POOL))]
        child2 = [1 if i in child2_indices else 0 for i in range(len(GENE_POOL))]

        return tuple(child1), tuple(child2)

    def _mutate(self, individual: BitVector, mutation_rate: float, num_genes: int) -> BitVector:
        bits = list(individual)
        for i in range(len(bits)):
            if random.random() < mutation_rate:
                bits[i] = 1 - bits[i]
        # Repair to ensure exactly num_genes bits set
        ones = [i for i, b in enumerate(bits) if b == 1]
        zeros = [i for i, b in enumerate(bits) if b == 0]
        if len(ones) > num_genes:
            for idx in random.sample(ones, len(ones) - num_genes):
                bits[idx] = 0
        elif len(ones) < num_genes:
            for idx in random.sample(zeros, num_genes - len(ones)):
                bits[idx] = 1
        return tuple(bits)

    def optimize(
        self,
        evaluator: FitnessEvaluator,
        meta_heuristic_params: dict[str, int | float]
    ) -> tuple[BitVector, Fitness, History]:
        num_genes = meta_heuristic_params['num_bands']
        population_size = meta_heuristic_params['population_size']
        num_generations = meta_heuristic_params['num_iterations']
        elitism_count = meta_heuristic_params['elitism_count']
        mutation_rate = meta_heuristic_params['mutation_rate']
        stagnation_mutation_rate = meta_heuristic_params['stagnation_mutation_rate']
        stagnation_threshold = meta_heuristic_params['stagnation_threshold']
        tournament_size = meta_heuristic_params['tournament_size']
        selection = meta_heuristic_params['selection_type']
        crossover_type = meta_heuristic_params['crossover_type']
        blend_alpha = meta_heuristic_params['blend_alpha']

        if num_genes > len(GENE_POOL):
            raise ValueError('num_genes cannot be larger than GENE_POOL size')

        population = self._create_population(population_size, num_genes)

        best_global_individual: BitVector = None  # type: ignore[assignment]
        best_global_fitness: float = float('-inf')
        history: List[Tuple[BitVector, Fitness]] = []

        stagnation_count = 0
        best_prev: Tuple[Fitness, BitVector] | None = None
        initial_mutation_rate = mutation_rate

        print("\n--- 🧬 Iniciando Optimización con Algoritmo Genetico ---")

        for generation in range(num_generations):
            population_fitness = Parallel(n_jobs=-1)(
                delayed(lambda individual: (individual, float(evaluator.evaluate(self._bitvector_to_genes(individual)))))(individual) for individual in population
            )

            population_fitness.sort(key=lambda x: x[1], reverse=True)

            best_gen_ind, best_gen_fit = population_fitness[0]
            if best_gen_fit > best_global_fitness:
                best_global_fitness = best_gen_fit
                best_global_individual = best_gen_ind
            best = self._bitvector_to_genes(best_gen_ind)
            print(f"Generation {generation+1}/{num_generations} | Best fitness: {best_gen_fit:.4f} | Best ind: {best}")
            history.append((best, best_gen_fit))

            current_best = (best_gen_fit, best_gen_ind)
            if best_prev is not None:
                if current_best == best_prev:
                    stagnation_count += 1
                elif current_best[0] > best_prev[0]:
                    stagnation_count = 0
            best_prev = current_best

            if stagnation_count >= stagnation_threshold:
                print("Stagnation reached, increasing mutation rate...")
                mutation_rate = stagnation_mutation_rate
                stagnation_count = 0
            elif mutation_rate == stagnation_mutation_rate and best_prev and best_prev[0] > history[-1][1]:
                mutation_rate = initial_mutation_rate

            next_generation: List[BitVector] = []

            for i in range(min(elitism_count, len(population_fitness))):
                next_generation.append(population_fitness[i][0])

            while len(next_generation) < population_size:
                if selection == 'roulette':
                    parent1 = self._roulette_wheel_selection(population_fitness)
                    parent2 = self._roulette_wheel_selection(population_fitness)
                else:
                    parent1 = self._tournament_selection(population_fitness, tournament_size)
                    parent2 = self._tournament_selection(population_fitness, tournament_size)

                if crossover_type == 'blx':
                    if not blend_alpha:
                        raise Exception('An alpha value should be provided for blend crossover')
                    child1, child2 = self._blx_crossover(parent1, parent2, num_genes, blend_alpha)
                else:
                    child1, child2 = self._single_point_crossover(parent1, parent2, num_genes)

                child1 = self._mutate(child1, mutation_rate, num_genes)
                child2 = self._mutate(child2, mutation_rate, num_genes)

                if child1 not in next_generation:
                    next_generation.append(child1)
                if len(next_generation) < population_size and child2 not in next_generation:
                    next_generation.append(child2)

            population = next_generation
        return self._bitvector_to_genes(best_global_individual), best_global_fitness, history
