# CS6316 Machine Learning-Final Project

Final project of UVa CS6316 Machine Learning.

## Requirements
1. numpy
2. pandas
3. scikit-learn
4. matplotlib
5. pytorch
## Part 1
Enter code directory
```python
cd code
``````
Run command below to get pca analysis
```
python pca_analysis.py
```
Run command below to get pca analysis
```
python LR_KNN_SVM.py
```

Run command below to get result of Decision Tree, Random Forest and Adaboost
```
python decision_tree.py
python random_forest.py
python adaboost.py
```

## Part 2
Enter code directory
```python
cd code
``````
Run all the codeblocks in part2.ipynb

## LR_KNN_SVM.py instruction
Feature:  
Data Preprocessing: Automatic conversion of categorical variables to numerical labels using Label Encoding.  
Model Training and Evaluation: Implements Grid Search for hyperparameter tuning and cross-validation for performance evaluation.  
Performance Metrics: Calculates various metrics like accuracy, precision, recall, F1 score, and ROC AUC.  
Visualization: Generates plots for analyzing the impact of different hyperparameters on model performance.  

Usage:  
Data Loading: Load the load_data(filepath) function, where filepath needs to contain the dataset.  
Data Analysis: Call the advanced_train_and_evaluate_with_gridsearch function with the dataset.  

Functions:  
generate_lr_params(): Generates parameter grid for Logistic Regression.  
generate_knn_params(): Generates parameter grid for K-Nearest Neighbors.  
generate_svm_params(): Generates parameter grid for Support Vector Machine.  
plot_single_param_curve(grid_search, title, param_name, best_params): Plots the performance curve for a single parameter.  
calculate_metrics(cv_results): Computes performance metrics from cross-validation results.  
print_results(classifier_name, best_params, metrics): Prints the best parameters and performance metrics.  
load_data(filepath): Loads data from a given file path.  

Note: 
Modify the file paths in the load_data calls at the bottom of the script to point to your datasets.

