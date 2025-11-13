Execution PSO example:   
```sh
python3 -m fig_ga_svm.main pso svm \
--means_file=/Users/israel/Projects/fig_ga_svm/db/means.csv \
--std_file=/Users/israel/Projects/fig_ga_svm/db/std.csv \
--num_particles=10 \
--num_iterations=1 \
--num_bands=3 \
--mutation_rate=1.2 \
--w_max=2 \
--w_min=0.5 \
--c1=0.9 \
--c2=0.4
```

Executions GA example:
```sh
python3 -m fig_ga_svm.main ga svm \
--means_file=/Users/israel/Projects/fig_ga_svm/db/means.csv \
--std_file=/Users/israel/Projects/fig_ga_svm/db/std.csv \
--population_size=10 \
--num_iterations=1 \
--num_bands=3 \
--mutation_rate=0.2 \
--stagnation_mutation_rate=0.2 \
--elitism_count=1 \
--stagnation_threshold=50 \
--tournament_size=5 \
--selection_type=roulette
```

Example 2:
```
docker run -e PYTHONUNBUFFERED=1 isra-pso python3 -m fig_ga_svm.main pso svm --num_particles=200 --num_iterations=650 --num_bands=5 --mutation_rate=0.2 --w=2.0 --c1_min=0.4 --c1_max=1.0 --c2_min=0.4 --c2_max=1.0
```