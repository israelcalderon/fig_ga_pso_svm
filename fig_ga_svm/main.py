from fig_ga_svm import data
from fig_ga_svm import evaluator
from fig_ga_svm.optimizers import pso

import argparse
import pprint


OPTIMIZERS  = {
    'pso': pso.PSOOptimizer
}


EVALUATORS = {
    'svm': evaluator.SVMEvaluator
}


def format_arguments(args: argparse.Namespace) -> dict:
    not_arguments = ['heuristic', 'evaluator']
    return {k: v for k, v in vars(args).items() if k not in not_arguments}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("heuristic", help="The meta heuristic to execute (ga, pso)")
    parser.add_argument("evaluator", help="The classifier to use (svm, rf, xgboost)")
    parser.add_argument("--num_particles", help="Number of particles", type=int)
    parser.add_argument("--num_iterations", help="Number of iterations/generations", type=int)
    parser.add_argument("--num_bands", help="Number of bands for individual", type=int)
    parser.add_argument("--mutation_rate", help="Mutation rate", type=float)
    parser.add_argument("--w_max", help="Initial inertia (favors global exploration)", type=float)
    parser.add_argument("--w_min", help="Final intertia (favors local exploration)", type=float)
    parser.add_argument("--c1", help="Cognitive coeficient (influences pbest)", type=float)
    parser.add_argument("--c2", help="Social coeficient (influence gbest)", type=float)
    
    parser.add_argument("--means_file",
                        help="means dataset location",
                        default="/usr/src/app/db/means.csv")
    
    parser.add_argument("--std_file",
                        help="standar deviation dataset location",
                        default="/usr/src/app/db/std.csv")

    args = parser.parse_args()
    
    Optimizer = OPTIMIZERS.get(args.heuristic)
    Evaluator = EVALUATORS.get(args.evaluator)
    
    if not Optimizer:
        raise Exception(f'optimizer {args.heuristic} not implemented')
    
    if not Evaluator:
        raise Exception(f'evaluator {args.evaluators} not implemented')

    data_manager = data.DataManager(args.means_file, args.std_file)
    evaluator = Evaluator(data_manager)
    optimizer_arguments = format_arguments(args)
    best_individual, best_fitness, history = Optimizer().optimize(evaluator,
                                                                  optimizer_arguments)
    
    print("\n--- 🎉 Algoritmo Finalizado 🎉 ---")
    print(f"🏆 Mejor combinación de bandas encontrada: {best_individual}")
    print(f"⭐ Valor de fitness (F1 Score): {best_fitness:.4f}")

    print("\n📈 Historial del mejor fitness por iteración:")
    pprint.pprint(history)

    print("\n📋 Argumentos usados para ejecutar la búsqueda:")
    pprint.pprint(optimizer_arguments)
