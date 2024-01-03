import itertools
import numpy as np
from utils import load_data
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
dataset1 = load_data("project3_dataset1.txt")
dataset2 = load_data("project3_dataset2.txt")

if True:
    for i, dataset in enumerate([dataset1, dataset2]):
        print(f"="*30, f"dataset{i}", "="*30)

        # parameter_space = {
        # 'n_estimators_options': [10, 50, 100],
        # 'max_features_options': ['sqrt', 'log2', None],
        # 'max_depth_options': [None, 10, 20, 30],
        # 'min_samples_split_options': [2, 5, 10]
        # }
        
        parameter_space = {
            'n_estimators': [50,  100, 150, 200],
            'learning_rate': [0.01, 0.1, 1],

            # 'n_estimators': [50],
            # 'learning_rate': [0.1],
            'base_estimator': [
                DecisionTreeClassifier(max_depth=2),
                SVC(probability=True, kernel='rbf'),
                # KNeighborsClassifier(),
                # LogisticRegression(),
                GaussianNB(),
            ]
        }

        n_estimators_list =  ['50', '100', '150', '200']
        learning_rate_list =  ['0.01', '0.1', '1']
        base_estimator_list = ['Decision Tree', 'SVM', 'GuassianNB']
        
        test_acc_list = []

        param_combinations = list(itertools.product(
            parameter_space['n_estimators'],
            parameter_space['learning_rate'],
            parameter_space['base_estimator'],
        ))
        best_acc = 0.0
        best_score = None


        for combination in param_combinations:
            print(combination)
            n_estimators, learning_rate, base_estimator = combination

            clf = AdaBoostClassifier(random_state=123,
                                     n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     estimator=base_estimator
                                     )
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1_score': make_scorer(f1_score),
                'roc_auc': make_scorer(roc_auc_score)
            }
            
            scores = cross_validate(clf, dataset["data"], dataset["target"], scoring=scoring, cv=10)
            test_acc_list.append(scores["test_accuracy"].mean() * 100)
            if scores["test_accuracy"].mean() >= best_acc:
                best_score = scores
                best_acc = scores["test_accuracy"].mean()
                best_param = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "base_estimator": base_estimator,
                }
        # plt.plot(min_samples_split_list, test_acc_list)
        # plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        # plt.ylabel("Test Accuracy")
        # plt.xlabel("Min Sample Split")
        # plt.title(f"Min Sample Split Analysis on Dataset {i+1}")
        # plt.show()
        for score in best_score:
            print(f"{score}: {best_score[score].mean()}")
        print(f"Best accuracy for dataset_{i}:", best_acc)
        print(f"Best param for dataset_{i}:", best_param)