from fig_ga_svm import data
from fig_ga_svm import evaluator
from fig_ga_svm.optimizers import pso, genetic_algorithm

import argparse
import pprint


OPTIMIZERS  = {
    'pso': pso.PSOOptimizer,
    'ga': genetic_algorithm.GeneticAlgorithmOptimizer
}


EVALUATORS = {
    'svm': evaluator.SVMEvaluator,
    'rf': evaluator.RFEvaluator
}


def to_optimizer_arguments(args: argparse.Namespace) -> dict:
    not_arguments = ['heuristic', 'evaluator']
    return {k: v for k, v in vars(args).items() if k not in not_arguments}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("heuristic", help="The meta heuristic to execute (ga, pso)")
    parser.add_argument("evaluator", help="The classifier to use (svm, rf, xgboost)")
    parser.add_argument("--batch_size",
                        help="How many times run the complete process",
                        default=1,
                        type=int)
    parser.add_argument("--results_path",
                        help="Directory to store results",
                        default="/logs")
    parser.add_argument("--num_particles", help="Number of particles", type=int)
    parser.add_argument("--num_iterations", help="Number of iterations/generations", type=int)
    parser.add_argument("--num_bands", help="Number of bands for individual", type=int)
    parser.add_argument("--mutation_rate", help="Mutation rate", type=float)
    parser.add_argument("--w_max",
                        help="Initial inertia (favors global exploration)",
                        type=float)
    parser.add_argument("--w_min",
                        help="Final intertia (favors local exploration)",
                        type=float)
    parser.add_argument("--w",
                        help="Intertia when strategy is static",
                        type=float)

    parser.add_argument("--c1_min",
                        help="Min cognitive coeficient (influences pbest)",
                        type=float)
    parser.add_argument("--c1_max",
                        help="Max cognitive coeficient (influences pbest)",
                        type=float)
    parser.add_argument("--c1",
                        help="Cognitive coeficient when static (influences pbest)",
                        type=float)

    parser.add_argument("--c2",
                        help="Social coeficient when static (influence gbest)",
                        type=float)
    parser.add_argument("--c2_min",
                        help="Min social coeficient (influence gbest)",
                        type=float)
    parser.add_argument("--c2_max",
                        help="Max social coeficient (influence gbest)",
                        type=float)

    parser.add_argument("--means_file",
                        help="means dataset location",
                        default="/usr/src/app/db/means.csv")
    parser.add_argument("--std_file",
                        help="standar deviation dataset location",
                        default="/usr/src/app/db/std.csv")
    # genetic
    parser.add_argument("--population_size",
                        help="Number of individuals per generation",
                        type=int)
    parser.add_argument("--elitism_count",
                        help="Number of best to pass to the next generation",
                        type=int)
    parser.add_argument("--stagnation_mutation_rate",
                        help="To deprecte: rate to use on stagnation",
                        type=float)
    parser.add_argument("--stagnation_threshold",
                        help="To deprecte: Treshold to trigger stagnation flag",
                        type=int)
    parser.add_argument("--tournament_size",
                        help="Size of tournament for tournament parent selection",
                        type=int)
    parser.add_argument("--selection_type",
                        help="Selection type to use: tournament or roulette")
    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    
    results_manager = data.ResultsManager()
    Optimizer = OPTIMIZERS.get(args.heuristic)
    Evaluator = EVALUATORS.get(args.evaluator)
    
    if not Optimizer:
        raise Exception(f'optimizer {args.heuristic} not implemented')
    
    if not Evaluator:
        raise Exception(f'evaluator {args.evaluators} not implemented')

    data_manager = data.DataManager(args.means_file, args.std_file)
    evaluator = Evaluator(data_manager)
    optimizer_arguments = to_optimizer_arguments(args)
    for i in range(args.batch_size):
        print(f"\n ⏱️ batch {i+1} of {args.batch_size}")
        best_individual, best_fitness, history = Optimizer().optimize(evaluator,
                                                                    optimizer_arguments)
        precise_fitness = evaluator.evaluate_precise(best_individual)
        
        print("\n--- 🎉 Algoritmo Finalizado 🎉 ---")
        print(f"🏆 Mejor combinación de bandas encontrada: {best_individual}")
        print(f"⭐ Valor de fitness (F1 Score): {best_fitness:.4f}")

        print("\n📈 Historial del mejor fitness por iteración:")
        pprint.pprint(history)

        print("\n📋 Argumentos usados para ejecutar la búsqueda:")
        pprint.pprint(optimizer_arguments)

        results_manager.store_results(args.results_path,
                                    args.heuristic,
                                    args.evaluator,
                                    best_individual,
                                    best_fitness,
                                    precise_fitness,
                                    history,
                                    optimizer_arguments)
