Execution example:   
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