import pprint

from dataclasses import dataclass
from typing import NoReturn

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fig_ga_svm import data
from fig_ga_svm import evaluator


@dataclass(frozen=True)
class Individual:
    description: str
    bands: tuple


BEST_INDIVIDUALS = [
    Individual('Genetic Algorithm 3 Bands', ('406.17', '981.98', '397.01',)),
    Individual('Genetic Algorithm 5 Bands', ('404.86', '763.94', '397.01', '707.93', '398.32',)),
    Individual('Genetic Algorithm 6 Bands', ('824.47', '554.24', '956.69', '398.32', '994.65', '397.01',)),
    Individual('PSO 3 Bands', ('397.01', '406.17', '980.57',)),
    Individual('PSO 5 Bands', ('1004.52', '397.01', '398.32', '404.86', '713.38',)),
    Individual('PSO 6 Bands', ('1004.52', '397.01', '407.48', '739.30', '931.48', '970.73',))
]


def generate_global_shap(model, X_train, X_test, feature_names, description) -> list:
    background_data = shap.sample(X_train, 50)
    
    # Standardize feature names to match the DataFrame columns if the provided ones mismatch
    if feature_names is None or len(feature_names) != X_train.shape[1]:
        feature_names = X_train.columns.tolist()

    # Define wrapper to avoid Pipeline feature_names_in_ setter AttributeError
    # and reconstruct DataFrame to avoid scikit-learn feature name warnings
    def predict_proba_wrapper(X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        return model.predict_proba(X)

    explainer = shap.KernelExplainer(predict_proba_wrapper, background_data)
    shap_values = explainer.shap_values(X_test)
    
    # Handle list of arrays (older SHAP version) vs 3D numpy array (newer SHAP version)
    if isinstance(shap_values, list):
        shap_values_positive_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3 and shap_values.shape[2] > 1:
            shap_values_positive_class = shap_values[:, :, 1]
        else:
            shap_values_positive_class = shap_values
    else:
        shap_values_positive_class = shap_values

    plt.figure(figsize=(10, 6))

    shap.summary_plot(shap_values_positive_class,
                      X_test,
                      feature_names=feature_names,
                      plot_type='dot',
                      show=False)
    plt.title(f'SHAP Global Analisys: Variable Importance {description}', fontsize=14, pad=15)
    plt.tight_layout()
    file_name = f'results_analisys/v1/{description}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.show()
    return shap_values_positive_class


def generate_bar_plot_shap(shap_values_positive_class, X_test, feature_names, description):
    if feature_names is None or len(feature_names) != X_test.shape[1]:
        feature_names = X_test.columns.tolist()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_positive_class, 
        X_test, 
        feature_names=feature_names, 
        plot_type="bar", 
        show=False,
        color="#2b8cbe"
    )
    plt.title(f'SHAP Mean importance of varaibles {description}', fontsize=14, pad=15)
    plt.xlabel('Mean absolute impact in classification (|SHAP Value|)', fontsize=12)
    plt.tight_layout()
    file_name = f'results_analisys/v1/SHAP_Bar_Plot_Importance {description}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')


def run(means_path: str, std_path: str) -> None:
    data_manager = data.DataManager(means_path, std_path, test_size=0.5)
    
    for individual in BEST_INDIVIDUALS:
        evaluator_ = evaluator.SVMEvaluator(data_manager)
        _f1_score, model = evaluator_.evaluate_precise(individual.bands, probability=True)
        shap_vals = generate_global_shap(
            model,
            data_manager.get_preprocessed_features(individual.bands),
            data_manager.get_preprocessed_features(individual.bands, True),
            list(individual.bands),
            individual.description)
        
        generate_bar_plot_shap(
            shap_vals,
            data_manager.get_preprocessed_features(individual.bands, True),
            list(individual.bands),
            individual.description)
