import itertools
import numpy as np
from utils import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
dataset1 = load_data("project3_dataset1.txt")
dataset2 = load_data("project3_dataset2.txt")

clf = RandomForestClassifier()

if True:
    for i, dataset in enumerate([dataset1, dataset2]):
        print(f"="*30, f"dataset{i}", "="*30)

        parameter_space = {
        'n_estimators_options': [10, 50, 100],
        'max_features_options': ['sqrt', 'log2', None],
        'max_depth_options': [None, 10, 20, 30],
        'min_samples_split_options': [2, 5, 10]
        }
        
        # parameter_space = {
        #     'n_estimators_options': [100],
        #     'max_features_options': [None],
        #     'max_depth_options': [10],
        #     'min_samples_split_options': [2, 3, 5, 8, 10, 15, 20, 30, 35, 40, 50, 60, 70]
        # }

        n_estimators_list =  [ '10', '30', '50', '80', '100', '150', '200', '250', '300']
        max_features_list = ["log2", 'sqrt'," None"]
        max_depth_list = ["None", "10", "20", "30", '40', '50', '60', '70', '80', '100']
        min_samples_split_list =  ['2','3', '5', '8', '10', '15', '20', '30', '35', '40', '50', '60', '70']
        test_acc_list = []

        param_combinations = list(itertools.product(
            parameter_space['n_estimators_options'],
            parameter_space['max_features_options'],
            parameter_space['max_depth_options'],
            parameter_space['min_samples_split_options'],
        ))
        best_acc = 0.0
        best_score = None


        for combination in param_combinations:
            n_estimators_option, max_features_option, max_depth_option, min_samples_split_option = combination

            clf = RandomForestClassifier(random_state=123,
                                         max_features=max_features_option,
                                         n_estimators=n_estimators_option,
                                         max_depth=max_depth_option,
                                         min_samples_split=min_samples_split_option
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
                    "n_estimators_option": n_estimators_option,
                    "max_features_option": max_features_option,
                    "max_depth_option": max_depth_option,
                    "min_samples_split_option": min_samples_split_option
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