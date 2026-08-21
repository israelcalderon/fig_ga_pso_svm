import numpy as np                                                                                                                                                                                                                                                                                     
import pandas as pd                                                                                                                                                                                                                                                                                    
from sklearn.decomposition import PCA                                                                                                                                                                                                                                                                  
from sklearn.metrics import f1_score                                                                                                                                                                                                                                                                   
from sklearn.model_selection import cross_val_predict                                                                                                                                                                                                                                                  
from sklearn.preprocessing import StandardScaler                                                                                                                                                                                                                                                       
from sklearn.svm import SVC                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                        
def main():
    # 1-3. Load datasets and add suffixes to feature columns
    df_means = pd.read_csv("db/means.csv")
    df_stds = pd.read_csv("db/std.csv")

    X_means = df_means.drop(columns=["class"]).add_suffix("_mean")
    X_stds = df_stds.drop(columns=["class"]).add_suffix("_std")

    # 4. Concatenate features horizontally (896 total features)
    X_combined = pd.concat([X_means, X_stds], axis=1)

    # 5. Extract target variable
    y = df_means["class"]

    # 6. Standardize feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # 7. Apply PCA to extract 5 principal components
    pca = PCA(n_components=5, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # 8 & 10. Initialize SVM using settings from SVMEvaluator (kernel='rbf', random_state=42, class_weight='balanced')
    svm = SVC(kernel="rbf", random_state=42, class_weight="balanced")

    # 9. 5-Fold cross-validation predictions
    y_pred = cross_val_predict(svm, X_pca, y, cv=5)

    # 11. Calculate and print overall F1-Score
    f1 = f1_score(y, y_pred, average="binary")
    print(f"Overall 5-Fold Cross-Validation F1-Score: {f1:.4f}\n")

    # 12. Extract PCA components_ and print top 5 influencing features for PC1 and PC2
    feature_names = X_combined.columns
    for i, pc_name in enumerate(["PC1", "PC2"]):
        abs_loadings = np.abs(pca.components_[i])
        top5_idx = np.argsort(abs_loadings)[::-1][:5]

        print(f"Top 5 features influencing {pc_name}:")
        for rank, idx in enumerate(top5_idx, 1):
            print(
                f"  {rank}. {feature_names[idx]} (Absolute Loading: {abs_loadings[idx]:.5f})"
            )
        print()


if __name__ == "__main__":
    main()
