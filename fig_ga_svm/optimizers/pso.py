from typing import Dict, Literal, Optional, Union
from joblib import Parallel, delayed

import numpy as np

from fig_ga_svm.evaluator import FitnessEvaluator
from fig_ga_svm.data import DataManager, GENE_POOL
from fig_ga_svm.optimizers.optimizer import Optimizer, Individual, Fitness, History


type PSOParams = Dict[Literal['num_particles', 'num_iterations', 'num_bands',
                              'mutation_rate', 'w_max', 'w_min', 'c1', 'c2'],
                      Union[float, int]]


class PSOOptimizer(Optimizer):

    def __init__(self) -> None:
        self.defaults = {
            'num_particles': 150, 
            'num_iterations': 200, 
            'num_bands': 3,
            'mutation_rate': 0.1, 
            'w_max': 2.0,
            'w_min': 2.0, 
            'c1': 0.9,
            'c2': 0.4}

    def __map_position_to_bands(self, position: np.ndarray) -> tuple:
        """
        Converts a vector con continous positions [0, 1] in a tuple of discrete bands
        It handle duplicated to ensure that the bands are unique
        """
        num_total_bands = len(GENE_POOL)
        indices = set()

        for value in position:
            index = int(value * (num_total_bands - 1)) # Escale index range
            # If the index is duplicated search for the next one izi pizi
            while index in indices:
                index = (index + 1) % num_total_bands
            indices.add(index)
        return tuple(sorted([GENE_POOL[i] for i in indices]))

    def __inertia(self,
                iteration: int,
                total_iterations: int,
                w: Optional[float] = None,
                w_min: Optional[float] = None,
                w_max: Optional[float] = None) -> float:
        if w and (w_min or w_max):
            raise Exception('Ambiguous inertia strategy')

        if w:
            return w

        if any([i is None for i in [w_min, w_max]]):
            raise Exception(f'For dynamic strategy w_min and w_max are required')
        return w_max - (w_max - w_min) * iteration / total_iterations

    def __coeficients(self, iteration, total_iterations, **kwargs) -> tuple[float, float]:
        c1 = kwargs.get('c1')
        c1_min = kwargs.get('c1_min')
        c1_max = kwargs.get('c1_max')

        c2 = kwargs.get('c2')
        c2_min = kwargs.get('c2_min')
        c2_max = kwargs.get('c2_max')

        if (c1 and (c1_min or c1_max) or (c2 and (c2_min or c2_max))):
            raise Exception('Abifuous coeficient strategy')
        
        if c1 and c2:
            return c1, c2
        
        c1_dynamic = c1_max - (c1_max - c1_min) * iteration / total_iterations
        c2_dynamic = c2_min + (c2_max - c2_min) * iteration / total_iterations
        return c1_dynamic, c2_dynamic

    def optimize(
            self,
            evaluator: FitnessEvaluator,
            meta_heuristic_params: dict) -> tuple[Individual, Fitness, History]:
        
        N_PARTICLES = meta_heuristic_params['num_particles']
        NUM_BANDS = meta_heuristic_params['num_bands']
        MUTATION_RATE = meta_heuristic_params['mutation_rate']
        N_ITERATIONS = meta_heuristic_params['num_iterations']
        
        # positions random vectors of size NUM_BANDS with values in [0, 1]
        particles_pos = np.random.rand(N_PARTICLES, NUM_BANDS)
        # velocity: vectors initialized at zero or small values
        particles_vel = np.zeros((N_PARTICLES, NUM_BANDS))
        # personal bests (pbest) init with current positions
        particles_pbest = np.copy(particles_pos)

        # Initial positions fitness
        pbest_fitness = np.array(
            Parallel(n_jobs=-1)(
                delayed(lambda pos: evaluator.evaluate(self.__map_position_to_bands(pos)))(pos) for pos in particles_pbest
            )
        )
        # Encontrar el mejor global (gbest)
        gbest_idx = np.argmax(pbest_fitness)
        gbest = particles_pbest[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]
        gbest_bands = self.__map_position_to_bands(gbest)
        history = []

        print(f"Mejor Fitness Inicial: {gbest_fitness:.4f} con bandas {gbest_bands}")
        print("\n--- 🧠 Iniciando Optimización ---")

        for i in range(N_ITERATIONS):
            w = self.__inertia(i,
                             N_ITERATIONS,
                             meta_heuristic_params['w'],
                             meta_heuristic_params['w_min'],
                             meta_heuristic_params['w_max'])
            
            C1, C2 = self.__coeficients(i, N_ITERATIONS, **meta_heuristic_params)
            
            # Vectorized random coefficients per particle
            r1 = np.random.rand(N_PARTICLES, NUM_BANDS)
            r2 = np.random.rand(N_PARTICLES, NUM_BANDS)

            # Cognitive and social components (gbest broadcasts across particles)
            cognitive = C1 * r1 * (particles_pbest - particles_pos)
            social = C2 * r2 * (gbest - particles_pos)

            # 1. Actualizar velocidades y posiciones de forma vectorizada
            particles_vel = w * particles_vel + cognitive + social
            particles_pos = particles_pos + particles_vel
            particles_pos = np.clip(particles_pos, 0.0, 1.0)
            mutation_mask = np.random.rand(N_PARTICLES, 1) < MUTATION_RATE

            new_random_positions = np.random.rand(N_PARTICLES, NUM_BANDS)
            particles_pos = np.where(mutation_mask,
                                    new_random_positions,
                                    particles_pos)

            # Evaluar fitness de todas las partículas en paralelo (gbest permanece fijo dentro de la iteración)
            fitness_results = Parallel(n_jobs=-1)(
                delayed(lambda pos: evaluator.evaluate(self.__map_position_to_bands(pos)))(pos)
                for pos in particles_pos
            )

            current_fitness_array = np.array(fitness_results)

            # Actualizar pbest donde corresponda
            improved_mask = current_fitness_array > pbest_fitness
            if np.any(improved_mask):
                particles_pbest[improved_mask] = particles_pos[improved_mask]
                pbest_fitness[improved_mask] = current_fitness_array[improved_mask]

            # Actualizar gbest usando los pbests actualizados
            new_gbest_idx = int(np.argmax(pbest_fitness))

            if pbest_fitness[new_gbest_idx] > gbest_fitness:
                gbest_fitness = pbest_fitness[new_gbest_idx]
                gbest = particles_pbest[new_gbest_idx]
                gbest_bands = self.__map_position_to_bands(gbest)

            history.append((gbest_bands, gbest_fitness))
            print(f"Iteración {i+1:03d}/{N_ITERATIONS} | Mejor Fitness Global: {gbest_fitness:.4f}")
        return gbest_bands, gbest_fitness, history
