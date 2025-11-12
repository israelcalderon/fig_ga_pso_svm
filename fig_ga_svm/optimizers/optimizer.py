from abc import ABC, abstractmethod
from typing import Iterable

from fig_ga_svm.evaluator import FitnessEvaluator


type Individual = tuple[str, ...]
type Fitness = float
type History = Iterable[tuple[Individual, Fitness]]


class Optimizer(ABC):

    @abstractmethod
    def optimize(
        self, 
        evaluator: FitnessEvaluator,
        meta_heuristic_params: dict[str, int | float]
        ) -> tuple[Individual, Fitness, History]:
        """
        Meta heuristic optimizer function, entry point to the optimization process
        
        :param evaluator: Evaluator with function to perform the evaluation/fitness function
        :param meta_heuristic_params: Meta heuristic specific parameters
        :return: Returns a tuple of the:
            - Individual: The best individual of the execution
            - Fitness: The fitness value of the this best individual
            - History: a List of all the individuals and their fitnesses in-
              secuential order
        :rtype: tuple[Individual, Fitness, History]
        """
        pass
