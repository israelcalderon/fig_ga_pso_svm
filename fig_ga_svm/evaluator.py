from abc import ABC, abstractmethod

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from fig_ga_svm.data import DataManager


class FitnessEvaluator(ABC):

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager

    @abstractmethod
    def evaluate(self, individual: tuple[str]) -> float:
        """
        Evaluates the the individual and returns the fitness function-
        punctuation (f1-score)
        
        :param individual: a individual which is a composition of N HS bands
        :rtype: float
        """
        pass


class SVMEvaluator(FitnessEvaluator):

    def evaluate(self, individual: tuple[str]) -> float:
        pipeline = make_pipeline(StandardScaler(), SVC(random_state=42, class_weight='balanced'))
        param_grid = {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': ['scale', 0.01, 0.1, 1],
            'svc__kernel': ['rbf', 'linear']
        }
        # IMPORTANT: when we parallelize over particles, keep GridSearchCV single-threaded
        # to avoid nested parallelism. Set n_jobs=1 here.
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=1)
        _, _, y = self.data_manager.get_training_data()
        X = self.data_manager.get_preprocessed_features(individual)
        grid.fit(X, y)
        return grid.best_score_
