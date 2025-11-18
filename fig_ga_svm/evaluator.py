from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from fig_ga_svm.data import DataManager


class FitnessEvaluator(ABC):

    def __init__(self, data_manager: DataManager) -> None:
        self.data_manager = data_manager

    @abstractmethod
    def evaluate(self, individual: tuple[str, ...]) -> float:
        """
        Evaluates the the individual and returns the fitness function-
        punctuation (f1-score)
        
        :param individual: a individual which is a composition of N HS bands
        :rtype: float
        """
        pass


class SVMEvaluator(FitnessEvaluator):

    def evaluate(self, individual: tuple[str, ...]) -> float:
        # pipeline = make_pipeline(StandardScaler(), SVC(random_state=42, class_weight='balanced'))
        pipeline = make_pipeline(
            StandardScaler(), 
            SVC(random_state=42, class_weight='balanced', kernel='rbf', C=1.0)
        )
        # param_grid = {
        #     'svc__C': [0.1, 1, 10, 100],
        #     'svc__gamma': ['scale', 0.01, 0.1, 1],
        #     'svc__kernel': ['rbf', 'linear']
        # }
        # IMPORTANT: when we parallelize over particles, keep GridSearchCV single-threaded
        # to avoid nested parallelism. Set n_jobs=1 here.
        # grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=1)
        _, _, y = self.data_manager.get_training_data()
        X = self.data_manager.get_preprocessed_features(individual)
        #return grid.best_score_
        try:
            # .ravel() se asegura que 'y' tenga el formato correcto
            f1_scores = cross_val_score(pipeline, X, y.values.ravel(), cv=3, scoring='f1', n_jobs=1)
            return f1_scores.mean()
        except Exception:
            # Esto puede pasar si un fold de CV no tiene ambas clases por casualidad
            return 0.0
    
    def evaluate_precise(self, individual: tuple[str, ...]) -> float:
        print("\n--- Iniciando Evaluación Precisa (SVM) ---")
        
        pipeline = make_pipeline(
            StandardScaler(), 
            SVC(random_state=42, class_weight='balanced')
        )

        param_grid = {
            'svc__C': [0.1, 1, 10, 100, 500, 1000],
            'svc__gamma': ['scale', 0.001, 0.01, 0.1, 1, 'auto'],
            'svc__kernel': ['rbf', 'linear']
        }
        grid = GridSearchCV(pipeline, 
                            param_grid, 
                            cv=5,            # <-- Más robusto
                            scoring='f1', 
                            n_jobs=-1,     # <-- Usa todos los cores
                            verbose=1)     # <-- Muestra el progreso

        print("Cargando datos de entrenamiento y prueba...")
        _, _, y_train = self.data_manager.get_training_data()
        X_train = self.data_manager.get_preprocessed_features(individual, use_test_set=False)
        
        _, _, y_test = self.data_manager.get_testing_data()
        X_test = self.data_manager.get_preprocessed_features(individual, use_test_set=True)
        
        print(f"Ajustando GridSearchCV en {X_train.shape[0]} muestras de entrenamiento...")
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_

        print("Evaluando en el conjunto de prueba...")
        y_pred = best_model.predict(X_test)
        
        final_f1_score = f1_score(y_test, y_pred, average='binary') 
        
        print("\n--- 🏆 Resultados de la Evaluación Precisa ---")
        print(f"Mejores Hiperparámetros (CV=5 en Train): {grid.best_params_}")
        print(f"Mejor F1-Score (CV=5 en Train): {grid.best_score_:.4f}")
        print(f"\nReporte de Clasificación en el CONJUNTO DE PRUEBA:")
        print(classification_report(y_test, y_pred))
        
        return final_f1_score


class RFEvaluator(FitnessEvaluator):

    def evaluate(self, individual: tuple[str, ...]) -> float:
        pipeline = make_pipeline(RandomForestClassifier(
            n_estimators=30,
            max_depth=10,
            min_samples_leaf=5,
            max_samples=0.6,
            random_state=42,
            class_weight='balanced',
            n_jobs=1
        ))
        _, _, y = self.data_manager.get_training_data()
        X = self.data_manager.get_preprocessed_features(individual)
        try:
            scores = cross_val_score(pipeline, X, y.values.ravel(), cv=3, scoring='f1', n_jobs=1)
            return scores.mean()
        except Exception:
            return 0.0

    def evaluate_precise(self, individual: tuple[str, ...]) -> float:
        """
        Entrena un modelo RF con una búsqueda de hiperparámetros
        EXHAUSTIVA (lenta) y lo evalúa en el CONJUNTO DE PRUEBA
        para obtener el F1-Score más preciso.
        """
        print("\n--- 🏁 Iniciando Evaluación Precisa (Random Forest) ---")
        
        # 1. Pipeline (Ahora podemos usar n_jobs=-1)
        # Random Forest NO necesita StandardScaler
        pipeline = make_pipeline(RandomForestClassifier(random_state=42,
                                                        class_weight='balanced',
                                                        n_jobs=-1)) # <-- Usa todos los cores

        # 2. Grilla de Parámetros EXTENSA
        param_grid = {
            'randomforestclassifier__n_estimators': [200, 400, 600],
            'randomforestclassifier__max_depth': [10, 20, 30, None],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4],
            'randomforestclassifier__min_samples_split': [2, 5, 10]
        }

        # 3. GridSearchCV PRECISO
        grid = GridSearchCV(pipeline, 
                            param_grid, 
                            cv=5,            # <-- Más robusto
                            scoring='f1', 
                            n_jobs=-1,     # <-- Usa todos los cores
                            verbose=1)     # <-- Muestra el progreso

        # 4. Obtener datos de ENTRENAMIENTO y PRUEBA
        print("Cargando datos de entrenamiento y prueba...")
        _, _, y_train = self.data_manager.get_training_data()
        X_train = self.data_manager.get_preprocessed_features(individual, use_test_set=False)
        
        _, _, y_test = self.data_manager.get_testing_data()
        X_test = self.data_manager.get_preprocessed_features(individual, use_test_set=True)
        
        # 5. Ajustar en el conjunto de ENTRENAMIENTO
        print(f"Ajustando GridSearchCV en {X_train.shape[0]} muestras de entrenamiento...")
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_

        # 6. Evaluar en el conjunto de PRUEBA
        print("Evaluando en el conjunto de prueba...")
        y_pred = best_model.predict(X_test)
        
        # 7. Calcular y reportar métricas
        final_f1_score = f1_score(y_test, y_pred, average='binary') # O 'weighted'
        
        print("\n--- 🏆 Resultados de la Evaluación Precisa (RF) ---")
        print(f"Mejores Hiperparámetros (CV=5 en Train): {grid.best_params_}")
        print(f"Mejor F1-Score (CV=5 en Train): {grid.best_score_:.4f}")
        print(f"\nReporte de Clasificación en el CONJUNTO DE PRUEBA:")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))        
        return final_f1_score
