import itertools
import numpy as np
from sklearn import tree
from utils import load_data
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dataset1 = load_data("project3_dataset1.txt")
dataset2 = load_data("project3_dataset2.txt")

clf = tree.DecisionTreeClassifier()

import matplotlib.pyplot as plt


if True:
    for i, dataset in enumerate([dataset1, dataset2]):
        # parameter_space = {
        #     'max_depths': [10],
        #     'min_samples_splits': [10],
        #     'min_samples_leafs':[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90],
        #     'criterions': ['entropy']
        #     }
        max_depth_list = ["None", "10", "20", "30", "40", "50", '60', '70', '80', '90', '100']
        min_samples_split_list = ["2", "5", "8", "10", '13', '15', '17', '20', '25', '30', '40', '50', '60', '70', '80', '90', '100', '150', '200']
        min_sample_leaf_list = ["2", "4",  '6', '8', '10', '12', '14', '16', '18', '20', '25', '30', '40', '50', '60', '70', '80', '90']
        test_acc_list = []
        print(f"="*30, f"dataset{i}", "="*30)

        parameter_space = {
        'max_depths': [None, 10, 20, 30, 40, 50],
        'min_samples_splits': [2, 5, 10],
        'min_samples_leafs': [1, 2, 4],
        'criterions': ['gini', 'entropy']
        }

        # parameter_space = {
        # 'max_depths': [None, 10, 20, 30, 40, 50],
        # 'min_samples_splits': [2, 5, 10],
        # 'min_samples_leafs': [1, 2, 4],
        # 'criterions': ['gini', 'entropy']
        # }

        param_combinations = list(itertools.product(
            parameter_space['max_depths'],
            parameter_space['min_samples_splits'],
            parameter_space['min_samples_leafs'],
            parameter_space['criterions'],
        ))
        print(param_combinations)
        best_acc = 0.0
        best_score = None
        for combination in param_combinations:
            max_depth, min_samples_split, min_samples_leaf, criterion = combination

            clf = tree.DecisionTreeClassifier(random_state=123, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

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
                best_acc = scores["test_accuracy"].mean()
                best_score = scores
                best_param = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "criterion": criterion
                }
        # plt.plot(min_sample_leaf_list, test_acc_list)
        # plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
        # plt.ylabel("Test Accuracy")
        # plt.xlabel("Min Sample Leaf")
        # plt.title(f"Min Sample Leaf Analysis on Dataset {i+1}")
        # plt.show()
        for score in best_score:
            print(f"{score}: {best_score[score].mean()}")
        print(f"Best accuracy for dataset_{i}:", best_acc)
        print(f"Best param for dataset_{i}:", best_param)