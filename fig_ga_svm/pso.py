import pandas as pd
import numpy as np
import random
import pprint
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# -----------------------------------------------------------------------------
# 1. POOL DE GENES Y PARÁMETROS DE PSO
# -----------------------------------------------------------------------------

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
N_PARTICLES = 250       # Número de partículas en el enjambre
N_ITERATIONS = 200     # Número de iteraciones
NUM_BANDS = 5          # Dimensiones del problema (bandas a seleccionar) 3

# --- Coeficientes de PSO ---
W_MAX = 0.9  # Inercia inicial (favorece exploración global)
W_MIN = 0.4  # Inercia final (favorece exploración local)
C1 = 2.5     # Coeficiente cognitivo (influencia de pbest)
C2 = 1.5     # Coeficiente social (influencia de gbest)

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
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_score_


def fitness_function(bands: tuple, X: pd.DataFrame, y: pd.DataFrame) -> float:
    """Función objetivo que evalúa una combinación de bandas."""
    if not bands:
        return 0.0
    columns = list(bands)
    x_subset = X[columns]
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

# -----------------------------------------------------------------------------
# 3. FUNCIÓN PRINCIPAL CON LÓGICA DE PSO
# -----------------------------------------------------------------------------

def main():
    """Carga los datos y ejecuta el algoritmo PSO optimizado."""
    # --- Carga de datos ---
    # !!! IMPORTANTE: Cambia esta ruta a la ubicación de tu archivo CSV !!!
    file_path = '/Users/israel/Projects/figs_means_db/db/figs_means.csv'
    df = pd.read_csv(file_path)
    X_train, _, y_train, _ = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.2, random_state=42, stratify=df['class'])

    # --- Inicialización del Enjambre ---
    print("--- 🚀 Inicializando Enjambre de Partículas ---")

    # Posiciones: Vectores aleatorios de tamaño NUM_BANDS con valores en [0, 1]
    particles_pos = np.random.rand(N_PARTICLES, NUM_BANDS)

    # Velocidades: Vectores inicializados a cero o valores pequeños
    particles_vel = np.zeros((N_PARTICLES, NUM_BANDS))

    # Mejores personales (pbest) inicializados con las posiciones actuales
    particles_pbest = np.copy(particles_pos)

    # Evaluar el fitness de las posiciones iniciales
    pbest_fitness = np.array([fitness_function(map_position_to_bands(pos), X_train, y_train) for pos in particles_pbest])

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

        for j in range(N_PARTICLES):
            r1, r2 = np.random.rand(2)

            # --- Ecuaciones clásicas de PSO ---
            # 1. Actualizar velocidad
            cognitive_vel = C1 * r1 * (particles_pbest[j] - particles_pos[j])
            social_vel = C2 * r2 * (gbest - particles_pos[j])
            particles_vel[j] = w * particles_vel[j] + cognitive_vel + social_vel

            # 2. Actualizar posición
            particles_pos[j] = particles_pos[j] + particles_vel[j]

            # 3. Mantener las posiciones dentro de los límites [0, 1]
            particles_pos[j] = np.clip(particles_pos[j], 0.0, 1.0)

            # --- Evaluación y actualización ---
            current_bands = map_position_to_bands(particles_pos[j])
            current_fitness = fitness_function(current_bands, X_train, y_train)

            # Actualizar pbest
            if current_fitness > pbest_fitness[j]:
                pbest_fitness[j] = current_fitness
                particles_pbest[j] = particles_pos[j]

                # Actualizar gbest
                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest = particles_pos[j]
                    gbest_bands = current_bands

        history.append(gbest_fitness)
        print(f"Iteración {i+1:03d}/{N_ITERATIONS} | Mejor Fitness Global: {gbest_fitness:.4f}")

    # --- Resultados Finales ---
    print("\n--- ✅ Algoritmo Finalizado ---")
    print(f"🏆 Mejor combinación de bandas encontrada: {gbest_bands}")
    print(f"⭐ Valor de fitness (F1 Score): {gbest_fitness:.4f}")

    print("\n📈 Historial del mejor fitness por iteración:")
    pprint.pprint(history)


if __name__ == '__main__':
    main()