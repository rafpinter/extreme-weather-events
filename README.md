# Extreme weather events - Kaggle competition 1

## Introduction

The prediction of extreme weather events is pivotal for understanding Earth's dynamics and for the early identification of hazardous conditions before they result in widespread devastation. In this competition, our objective was to construct machine learning models capable of distinguishing tropical cyclones and atmospheric rivers from typical background conditions using data from the ClimateNet dataset. To this end, three distinct types of models were developed: logistic regression, support vector machine (SVM), and LightGBM. The logistic regression model, crafted by the author, demonstrated an accuracy of 0.781/0.771 on the test set (private/public scores). The LightGBM model surpassed this with an accuracy of 0.784/0.777, while the SVM model posted an accuracy of 0.708/0.701. The latter two models were implemented using the LightGBM Python API and Scikit-learn, respectively. Although the results were encouraging, there remains significant scope for enhancement, particularly through meticulous refinement of feature selection and hyperparameter tuning.

## How to run the code

### Logistic regression

To run any code from the logistic regression, the "preprocess.py" and "logistic_regression.py" files need to be in the same directory. Furthermore, please adjust the path for your version of the "train.csv" and "test.csv" files. The main code for this part is the notebook "logistic_regression/log_reg_best_model.ipynb". To run this notebook, all you need to do is adjust the paths of the data. 

### Other models

For the LightGBM model, the file "preprocess.py" needs to be in the same directory. Please adjust the "train.csv" and "test.csv" files as well. The main notebook for this part is "other_models/LightGBM/lightgbm_best_model.ipynb". To run this notebook, all you need to do is adjust the paths of the data.

For the SVM files, please adjust the train and test dataset paths as well.

## Structure of the folder

```
data_exploratory_analysis/
logistic_regression/
    |_ other_tests/                                                             # Folder that contaings tests for Logistic Regression
        |_ log_reg.ipynb
        |_ log_reg_embedded_method.ipynb
        |_ log_reg_mode_features.ipynb
        |_ log_reg_optimized.ipynb
        |_ log_reg_optimized_balance.ipynb
        |_ log_reg_optimized_cross_valid_all_data_feateng.ipynb
        |_ log_reg_optimized_cross_valid_all_data_feateng_temporal_split.ipynb
        |_ log_reg_optimized_cross_valid_all_data_rm_dup.ipynb
        |_ log_reg_optimized_cross_valid_all_data_rm_out.ipynb
        |_ log_reg_optimized_cross_valid_feature_selection_after_best_model.ipynb
        |_ log_reg_optimized_cross_valid_report.ipynb
    |_ log_reg_best_model.ipynb                                                 # Code for the best submission of Logistic Regression                 
    |_ logistic_regression.py                                                   # Implementation of the Logistic Regression
    |_ preprocess.py                                                            # Preprocessing code used for the logistic regression
other_models/
    |_ LightGBM/
        |_ other_tests/                                                         # Has the code for other tests for LightGBM
            |_ lightgbm_cross_valid_all_feat_eng_cols.ipynb
            |_ lightgbm_cross_valid_original_cols_limit_tree_depth.ipynb
            |_ lightgbm_cross_valid_original_cols.ipynb
            |_ lightgbm_cross_valid.ipynb
            |_ lightgbm_temporal.ipynb
        |_ lightgbm_best_model.ipynb                                           # Code for the best submission of Logistic Regression
        |_ preprocess.py                                                       # Preprocessing code used for the LightGBM (similar to Logistic Regression)
    |_ SVM/
        |_ svm_grid_search_cross_val.ipynb                                     # Code for the SVM test
requirements.txt
```