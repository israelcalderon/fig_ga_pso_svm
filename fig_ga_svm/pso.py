import pandas as pd
import numpy as np
import pprint
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from joblib import Parallel, delayed


GENE_POOL = [
    "397.01", "398.32", "399.63", "400.93", "402.24", "403.55", "404.86", "406.17", "407.48", "408.79", "410.10", "411.41", "412.72", "414.03", "415.34", "416.65", "417.96", "419.27", "420.58", "421.90", "423.21", "424.52", "425.83",
    "427.15", "428.46", "429.77", "431.09", "432.40", "433.71", "435.03", "436.34", "437.66", "438.97", "440.29", "441.60", "442.92", "444.23", "445.55", "446.87", "448.18", "449.50", "450.82", "452.13", "453.45", "454.77", "456.09", "457.40",
    "458.72", "460.04", "461.36", "462.68", "464.00", "465.32", "466.64", "467.96", "469.28", "470.60", "471.92", "473.24", "474.56", "475.88", "477.20", "478.52", "479.85", "481.17", "482.49", "483.81", "485.14", "486.46", "487.78", "489.11",
    "490.43", "491.75", "493.08", "494.40", "495.73", "497.05", "498.38", "499.70", "501.03", "502.35", "503.68", "505.01", "506.33", "507.66", "508.99", "510.31", "511.64", "512.97", "514.30", "515.63", "516.95", "518.28", "519.61", "520.94",
    "522.27", "523.60", "524.93", "526.26", "527.59", "528.92", "530.25", "531.58", "532.91", "534.25", "535.58", "536.91", "538.24", "539.57", "540.91", "542.24", "543.57", "544.90", "546.24", "547.57", "548.91", "550.24", "551.57", "552.91",
    "554.24", "555.58", "556.91", "558.25", "559.59", "560.92", "562.26", "563.59", "564.93", "566.27", "567.61", "568.94", "570.28", "571.62", "572.96", "574.30", "575.63", "576.97", "578.31", "579.65", "580.99", "582.33", "583.67", "585.01",
    "586.35", "587.69", "589.03", "590.37", "591.71", "593.06", "594.40", "595.74", "597.08", "598.42", "599.77", "601.11", "602.45", "603.80", "605.14", "606.48", "607.83", "609.17", "610.52", "611.86", "613.21", "614.55", "615.90", "617.24",
    "618.59", "619.94", "621.28", "622.63", "623.98", "625.32", "626.67", "628.02", "629.37", "630.71", "632.06", "633.41", "634.76", "636.11", "637.46", "638.81", "640.16", "641.51", "642.86", "644.21", "645.56", "646.91", "648.26", "649.61",
    "650.96", "652.31", "653.67", "655.02", "656.37", "657.72", "659.08", "660.43", "661.78", "663.14", "664.49", "665.84", "667.20", "668.55", "669.91", "671.26", "672.62", "673.97", "675.33", "676.68", "678.04", "679.40", "680.75", "682.11",
    "683.47", "684.82", "686.18", "687.54", "688.90", "690.25", "691.61", "692.97", "694.33", "695.69", "697.05", "698.41", "699.77", "701.13", "702.49", "703.85", "705.21", "706.57", "707.93", "709.29", "710.65", "712.02", "713.38", "714.74",
    "716.10", "717.47", "718.83", "720.19", "721.56", "722.92", "724.28", "725.65", "727.01", "728.38", "729.74", "731.11", "732.47", "733.84", "735.20", "736.57", "737.93", "739.30", "740.67", "742.03", "743.40", "744.77", "746.14", "747.50",
    "748.87", "750.24", "751.61", "752.98", "754.35", "755.72", "757.09", "758.46", "759.83", "761.20", "762.57", "763.94", "765.31", "766.68", "768.05", "769.42", "770.79", "772.17", "773.54", "774.91", "776.28", "777.66", "779.03", "780.40",
    "781.78", "783.15", "784.52", "785.90", "787.27", "788.65", "790.02", "791.40", "792.77", "794.15", "795.52", "796.90", "798.28", "799.65", "801.03", "802.41", "803.78", "805.16", "806.54", "807.92", "809.30", "810.67", "812.05", "813.43",
    "814.81", "816.19", "817.57", "818.95", "820.33", "821.71", "823.09", "824.47", "825.85", "827.23", "828.61", "830.00", "831.38", "832.76", "834.14", "835.53", "836.91", "838.29", "839.67", "841.06", "842.44", "843.83", "845.21", "846.59",
    "847.98", "849.36", "850.75", "852.13", "853.52", "854.91", "856.29", "857.68", "859.06", "860.45", "861.84", "863.23", "864.61", "866.00", "867.39", "868.78", "870.16", "871.55", "872.94", "874.33", "875.72", "877.11", "878.50", "879.89",
    "881.28", "882.67", "884.06", "885.45", "886.84", "888.23", "889.63", "891.02", "892.41", "893.80", "895.19", "896.59", "897.98", "899.37", "900.77", "902.16", "903.55", "904.95", "906.34", "907.74", "909.13", "910.53", "911.92", "913.32",
    "914.71", "916.11", "917.50", "918.90", "920.30", "921.69", "923.09", "924.49", "925.89", "927.28", "928.68", "930.08", "931.48", "932.88", "934.28", "935.68", "937.08", "938.48", "939.88", "941.28", "942.68", "944.08", "945.48", "946.88",
    "948.28", "949.68", "951.08", "952.48", "953.89", "955.29", "956.69", "958.09", "959.50", "960.90", "962.30", "963.71", "965.11", "966.52", "967.92", "969.33", "970.73", "972.14", "973.54", "974.95", "976.35", "977.76", "979.16", "980.57",
    "981.98", "983.38", "984.79", "986.20", "987.61", "989.02", "990.42", "991.83", "993.24", "994.65", "996.06", "997.47", "998.88", "1000.29", "1001.70", "1003.11", "1004.52"
]


# --- Parámetros de PSO ---
N_PARTICLES = 250      # Número de partículas en el enjambre
N_ITERATIONS = 650     # Número de iteraciones
NUM_BANDS = 5          # Dimensiones del problema (bandas a seleccionar) 3
MUTATION_RATE = 0.2

# --- Coeficientes de PSO ---
W_MAX = 0.95  # Inercia inicial (favorece exploración global)
W_MIN = 0.35  # Inercia final (favorece exploración local)
C1 = 1.9     # Coeficiente cognitivo (influencia de pbest)
C2 = 2.1     # Coeficiente social (influencia de gbest)

# -----------------------------------------------------------------------------
# 2. FUNCIÓN DE FITNESS Y MAPEADO
# -----------------------------------------------------------------------------

#def svm(X, y) -> float:
#    """Evalúa un conjunto de características usando SVM con validación cruzada."""
#    svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=42))
#    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#    f1_scores = cross_val_score(svm_clf, X, y, cv=cv, scoring='f1')
#    return np.mean(f1_scores)


def svm(X, y) -> float:
    pipeline = make_pipeline(StandardScaler(), SVC(random_state=42, class_weight='balanced'))
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 0.01, 0.1, 1],
        'svc__kernel': ['rbf', 'linear']
    }
    # IMPORTANT: when we parallelize over particles, keep GridSearchCV single-threaded
    # to avoid nested parallelism. Set n_jobs=1 here.
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=1)
    grid.fit(X, y)
    return grid.best_score_


def fitness_function(bands: tuple, X_means: pd.DataFrame, X_std: pd.DataFrame, y: pd.DataFrame) -> float:
    """Función objetivo que evalúa una combinación de bandas."""
    if not bands:
        return 0.0
    columns = list(bands)
    x_mean_subset = X_means[columns].add_suffix('_mean')
    x_std_subset = X_std[columns].add_suffix('_std')
    x_subset = pd.concat([x_mean_subset, x_std_subset], axis=1)
    return svm(x_subset, y)

def map_position_to_bands(position: np.ndarray) -> tuple:
    """
    Convierte un vector de posición continua [0,1] en una tupla de bandas discretas.
    Maneja duplicados para asegurar que las 3 bandas sean únicas.
    """
    num_total_bands = len(GENE_POOL)
    indices = set()

    for value in position:
        index = int(value * (num_total_bands - 1)) # Escalar al rango de índices

        # Mecanismo simple de reparación: si el índice está duplicado,
        # busca el siguiente disponible.
        while index in indices:
            index = (index + 1) % num_total_bands
        indices.add(index)

    return tuple(sorted([GENE_POOL[i] for i in indices]))


def validate_on_test_set(best_bands: tuple, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Entrena el modelo final con las mejores bandas y todos los datos de entrenamiento,
    y lo evalúa en el conjunto de prueba.
    """
    print("\n" + "="*60)
    print("--- 🧪 Iniciando Validación Final con el Conjunto de Prueba ---")
    print("="*60)

    # 1. Filtrar los datos para usar solo las mejores bandas
    X_train_best = X_train[list(best_bands)]
    X_test_best = X_test[list(best_bands)]

    # 2. Usar GridSearchCV para encontrar el mejor modelo con los datos de entrenamiento
    print("\nAjustando el modelo final con los mejores hiperparámetros...")
    pipeline = make_pipeline(StandardScaler(), SVC(random_state=42, class_weight='balanced'))
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 0.01, 0.1, 1],
        'svc__kernel': ['rbf', 'linear']
    }
    # Usamos más splits (cv=5) para una validación final más robusta
    # Use n_jobs=1 here too to avoid nested parallelism when evaluating particles
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=1)
    grid.fit(X_train_best, y_train)

    best_model = grid.best_estimator_
    print(f"\nMejores parámetros encontrados: {grid.best_params_}")

    # 3. Realizar predicciones en el conjunto de prueba
    y_pred = best_model.predict(X_test_best)

    # 4. Generar y mostrar métricas de evaluación
    print("\n" + "-"*60)
    print("--- 📊 Resultados de la Validación ---")
    print("-"*60)

    # Reporte de Clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Métricas adicionales
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"F1-Score (Ponderado) en Test: {f1:.4f}")
    print(f"Accuracy en Test: {accuracy:.4f}")

    # 5. Generar y guardar la Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.title('Matriz de Confusión en el Conjunto de Prueba')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.savefig('confusion_matrix.png')
    print("\nMatriz de confusión guardada como 'confusion_matrix.png'")
    plt.show()

# -----------------------------------------------------------------------------
# 3. FUNCIÓN PRINCIPAL CON LÓGICA DE PSO
# -----------------------------------------------------------------------------

def main():
    """Carga los datos y ejecuta el algoritmo PSO optimizado."""
    # file_means = '/usr/src/app/db/means.csv'
    # file_std = '/usr/src/app/db/std.csv'
    file_means = '/Users/israel/Projects/fig_ga_svm/db/means.csv'
    file_std = '/Users/israel/Projects/fig_ga_svm/db/std.csv'
    df_means = pd.read_csv(file_means)
    df_std = pd.read_csv(file_std)
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        df_means.drop('class', axis=1),
        df_std.drop('class', axis=1),
        df_means['class'],
        test_size=0.2,
        random_state=42,
        stratify=df_means['class'])
    
    stagnation_counter = 0
    global MUTATION_RATE

    print("--- 🚀 Inicializando Enjambre de Partículas ---")

    # Posiciones: Vectores aleatorios de tamaño NUM_BANDS con valores en [0, 1]
    particles_pos = np.random.rand(N_PARTICLES, NUM_BANDS)

    # Velocidades: Vectores inicializados a cero o valores pequeños
    particles_vel = np.zeros((N_PARTICLES, NUM_BANDS))

    # Mejores personales (pbest) inicializados con las posiciones actuales
    particles_pbest = np.copy(particles_pos)

    # Evaluar el fitness de las posiciones iniciales (paralelizado)
    # Use joblib.Parallel to compute fitness for each particle in parallel.
    pbest_fitness = np.array(
        Parallel(n_jobs=-1)(
            delayed(lambda pos: fitness_function(map_position_to_bands(pos), X1_train, X2_train, y_train))(pos)
            for pos in particles_pbest
        )
    )

    # Encontrar el mejor global (gbest)
    gbest_idx = np.argmax(pbest_fitness)
    gbest = particles_pbest[gbest_idx]
    gbest_fitness = pbest_fitness[gbest_idx]
    gbest_bands = map_position_to_bands(gbest)

    print(f"Mejor Fitness Inicial: {gbest_fitness:.4f} con bandas {gbest_bands}")

    # --- Bucle Principal de PSO ---
    print("\n--- 🧠 Iniciando Optimización ---")
    history = []


    for i in range(N_ITERATIONS):
        # Inercia decreciente: balancea exploración y explotación
        w = W_MAX - (W_MAX - W_MIN) * i / N_ITERATIONS

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
            delayed(lambda pos: fitness_function(map_position_to_bands(pos), X1_train, X2_train, y_train))(pos)
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

        # Set update the stagnation counter + increase mutation rate
        if pbest_fitness[new_gbest_idx] == gbest_fitness:
            stagnation_counter += 1

        if pbest_fitness[new_gbest_idx] > gbest_fitness:
            gbest_fitness = pbest_fitness[new_gbest_idx]
            gbest = particles_pbest[new_gbest_idx]
            gbest_bands = map_position_to_bands(gbest)

        history.append(gbest_fitness)
        print(f"Iteración {i+1:03d}/{N_ITERATIONS} | Mejor Fitness Global: {gbest_fitness:.4f}")

    # --- Resultados Finales ---
    print("\n--- ✅ Algoritmo Finalizado ---")
    print(f"🏆 Mejor combinación de bandas encontrada: {gbest_bands}")
    print(f"⭐ Valor de fitness (F1 Score): {gbest_fitness:.4f}")

    print("\n📈 Historial del mejor fitness por iteración:")
    pprint.pprint(history)
    # validate_on_test_set(gbest_bands, X1_train, y_train, X1_test, y_test)


if __name__ == '__main__':
    main()