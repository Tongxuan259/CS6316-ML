import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Define functions for generating parameter grids
def generate_lr_params():
    C_values = [0.1, 1, 10]
    max_iter_values = [500, 1000, 2000]
    solvers = ['lbfgs', 'sag']
    return [{'C': [c], 'max_iter': [max_iter], 'solver': [solver]}
            for c in C_values for max_iter in max_iter_values for solver in solvers]


def generate_knn_params():
    n_neighbors_values = [3, 5, 7]
    weights_options = ['uniform', 'distance']
    return [{'n_neighbors': [n], 'weights': [w]}
            for n in n_neighbors_values for w in weights_options]


def generate_svm_params():
    C_values = [0.1, 1, 10]
    kernels = ['linear', 'rbf', 'poly']
    return [{'C': [c], 'kernel': [k]}
            for c in C_values for k in kernels]


# Plotting function
def plot_single_param_curve(grid_search, title, param_name, best_params):
    param_template = {k: best_params[k] for k in best_params if k != param_name}

    unique_values = np.unique([params[param_name] for params in grid_search.cv_results_['params']])

    test_scores = []
    for val in unique_values:
        for i, params in enumerate(grid_search.cv_results_['params']):
            if params[param_name] == val and all(params[k] == param_template[k] for k in param_template):
                test_scores.append(grid_search.cv_results_['mean_test_score'][i])

    plt.figure()
    plt.title(f"{title}: {param_name} Analysis")
    plt.xlabel(f"{param_name} Values")
    plt.ylabel("Test Accuracy")
    plt.plot(unique_values, test_scores, marker='o')
    plt.grid()
    plt.show()


# Calculate metrics
def calculate_metrics(cv_results):
    metrics = {
        'Accuracy': np.mean(cv_results['test_accuracy']),
        'Precision': np.mean(cv_results['test_precision']),
        'Recall': np.mean(cv_results['test_recall']),
        'F1 Score': np.mean(cv_results['test_f1_score']),
        'AUC': np.mean(cv_results['test_roc_auc'])
    }
    return metrics


# Print results
def print_results(classifier_name, best_params, metrics):
    print(f"---- {classifier_name} Results ----")
    print("Best Params:", best_params)
    for metric, score in metrics.items():
        print(f"{metric}: {score}")
    print("--------------------------------------")


# Main function for training and evaluation
def advanced_train_and_evaluate_with_gridsearch(X, y, dataset_name):
    # Convert categorical variables to numerical labels if needed
    label_encoders = {}
    for i in range(X.shape[1]):
        if isinstance(X[0, i], str):
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i])
            label_encoders[i] = le

    print(f"---- {dataset_name} ----")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1_score': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

    # Logistic Regression
    lr_params = generate_lr_params()
    lr = LogisticRegression()
    grid_search_lr = GridSearchCV(lr, lr_params, cv=10, scoring='accuracy', return_train_score=True)
    grid_search_lr.fit(X_train, y_train)
    best_lr = LogisticRegression(**grid_search_lr.best_params_)
    cv_results_lr = cross_validate(best_lr, X, y, cv=10, scoring=scoring)
    print_results("Logistic Regression", grid_search_lr.best_params_, calculate_metrics(cv_results_lr))

    plot_single_param_curve(grid_search_lr, "Logistic Regression", 'C', grid_search_lr.best_params_)
    plot_single_param_curve(grid_search_lr, "Logistic Regression", 'max_iter', grid_search_lr.best_params_)
    plot_single_param_curve(grid_search_lr, "Logistic Regression", 'solver', grid_search_lr.best_params_)

    # KNN
    knn_params = generate_knn_params()
    knn = KNeighborsClassifier()
    grid_search_knn = GridSearchCV(knn, knn_params, cv=10, scoring='accuracy', return_train_score=True)
    grid_search_knn.fit(X_train, y_train)
    best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
    cv_results_knn = cross_validate(best_knn, X, y, cv=10, scoring=scoring)
    print_results("K-NN", grid_search_knn.best_params_, calculate_metrics(cv_results_knn))

    plot_single_param_curve(grid_search_knn, "KNN", 'n_neighbors', grid_search_knn.best_params_)
    plot_single_param_curve(grid_search_knn, "KNN", 'weights', grid_search_knn.best_params_)

    # SVM
    svm_params = generate_svm_params()
    svm = SVC(probability=True)  # for SVM, probability for AUC
    grid_search_svm = GridSearchCV(svm, svm_params, cv=10, scoring='accuracy', return_train_score=True)
    grid_search_svm.fit(X_train, y_train)
    best_svm = SVC(**grid_search_svm.best_params_, probability=True)
    cv_results_svm = cross_validate(best_svm, X, y, cv=10, scoring=scoring)
    print_results("SVM", grid_search_svm.best_params_, calculate_metrics(cv_results_svm))

    plot_single_param_curve(grid_search_svm, "SVM", 'C', grid_search_svm.best_params_)
    plot_single_param_curve(grid_search_svm, "SVM", 'kernel', grid_search_svm.best_params_)


# Load and prepare datasets
def load_data(filepath):
    df = pd.read_csv(filepath, delimiter='\t', header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


# Load datasets
X1, y1 = load_data(r'G:\UVA\CS 6316\proj\project3_dataset1.txt')
X2, y2 = load_data(r'G:\UVA\CS 6316\proj\project3_dataset2.txt')

# Run analysis
advanced_train_and_evaluate_with_gridsearch(X1, y1, "Dataset1")
advanced_train_and_evaluate_with_gridsearch(X2, y2, "Dataset2")
