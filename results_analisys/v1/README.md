# Mean reflectance + Std deviation of mean reflectance

This execution was made with the databases:
- db/means.csv
- db/std.csv

The algorithms runned were:
- Genetic Algorithm
- Particle Swarm Optimization

The individuals tested were 3, 5 and 6 bands. Therefore the vector consisted of:
- 3 Bands individual = [[mean, standar deviation], [mean, standar deviation], [mean, standar deviation]]
- 5 Bands individual = [[mean, standar deviation], [mean, standar deviation], [mean, standar deviation], [mean, standar deviation], [mean, standar deviation]]
- 6 Bands individual = [[mean, standar deviation], [mean, standar deviation], [mean, standar deviation], [mean, standar deviation], [mean, standar deviation], [mean, standar deviation]]

The classifier used with the best individual of the meta-heuristic  was a Support Vector Machine with the following parameters:
```python
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
                        cv=5,
                        scoring='f1',
                        n_jobs=-1,
                        verbose=1)
```
## Individuals to evaluate according to the results of optimization
The following individuals were found running the mentioned algorithm with the parameters documented at the csv log respective file, in the table is reported the description of the individual, the bands found for the individual and the F1-Score obtained from the evaluate_precise function during the optimization process.

### Genetic Algorithm
### Mean TODO
The corresponding log files are:
- results_analisys/v1/ga_svm_3b_2025_11_16.csv
- results_analisys/v1/ga_svm_5b_2025_11_16.csv
- results_analisys/v1/ga_svm_6b_2025_11_15.csv

|Description|Bands                                         |F1-Score          |
|-----------|----------------------------------------------|------------------|
|3 Bands    |                        |0.|
|5 Bands    |        |0.|
|6 Bands    ||0.|

#### Mean + Standard deviation
The corresponding log files are:
- results_analisys/v1/ga_svm_3b_2025_11_16.csv
- results_analisys/v1/ga_svm_5b_2025_11_16.csv
- results_analisys/v1/ga_svm_6b_2025_11_15.csv

|Description|Bands                                         |F1-Score          |
|-----------|----------------------------------------------|------------------|
|3 Bands    |406.17, 981.98, 397.01                        |0.8173913043478261|
|5 Bands    |404.86, 763.94, 397.01, 707.93, 398.32        |0.8360655737704918|
|6 Bands    |824.47, 554.24, 956.69, 398.32, 994.65, 397.01|0.8205128205128205|

### PSO
The corresponding log files are:
- results_analisys/v1/pso_svm_3b_2025_11_17.csv
- results_analisys/v1/pso_svm_5b_2025_11_16.csv
- results_analisys/v1/pso_svm_6b_2025_11_15.csv

|Description|Bands                                          |F1-Score          |
|-----------|-----------------------------------------------|------------------|
|3 Bands    |397.01, 406.17, 980.57                         |0.8070175438596491|
|5 Bands    |1004.52, 397.01, 398.32, 404.86, 713.38        |0.7627118644067796|
|6 Bands    |1004.52, 397.01, 407.48, 739.30, 931.48, 970.73|0.8214285714285714|

## Analisys
The results of this excecution will be analyzed with a SHAP Beeswarm graph and a mean importance Bar Plot. For each of the individuals a dataset will be created containing only the selected bands by each individual. 

## PCA Analysis
A PCA Analysis was runned with 5 PCA to match the best results from the 5 bands found by the meta heuristics and the results are th following:
```
Overall 5-Fold Cross-Validation F1-Score: 0.6484

Top 5 features influencing PC1:
  1. 450.82_mean (Absolute Loading: 0.04504)
  2. 452.13_mean (Absolute Loading: 0.04501)
  3. 449.50_mean (Absolute Loading: 0.04500)
  4. 445.55_mean (Absolute Loading: 0.04499)
  5. 446.87_mean (Absolute Loading: 0.04498)

Top 5 features influencing PC2:
  1. 805.16_mean (Absolute Loading: 0.04961)
  2. 806.54_mean (Absolute Loading: 0.04961)
  3. 814.81_mean (Absolute Loading: 0.04961)
  4. 803.78_mean (Absolute Loading: 0.04961)
  5. 802.41_mean (Absolute Loading: 0.04960)
```