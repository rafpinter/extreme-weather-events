# Extreme weather events - Kaggle competition 1

## Introduction

The prediction of extreme weather events is pivotal for understanding Earth's dynamics and for the early identification of hazardous conditions before they result in widespread devastation. In this competition, our objective was to construct machine learning models capable of distinguishing tropical cyclones and atmospheric rivers from typical background conditions using data from the ClimateNet dataset. To this end, three distinct types of models were developed: logistic regression, support vector machine (SVM), and LightGBM. The logistic regression model, crafted by the author, demonstrated an accuracy of 0.781/0.771 on the test set (private/public scores). The LightGBM model surpassed this with an accuracy of 0.784/0.777, while the SVM model posted an accuracy of 0.708/0.701. The latter two models were implemented using the LightGBM Python API and Scikit-learn, respectively. Although the results were encouraging, there remains significant scope for enhancement, particularly through meticulous refinement of feature selection and hyperparameter tuning.

## Structure of the folder

```
data_exploratory_analysis/
logistic_regression/
    |_ other_tests/
        |_ log_reg.ipynb
        |_ log_reg_2.ipynb
        |_ log_reg_2_augmented.ipynb
        |_ log_reg_2_optimize_with_augmented.ipynb
        |_ log_reg_2_optimized.ipynb
        |_ log_reg_2_optimized_balance.ipynb
        |_ log_reg_2_optimized_cross_valid.ipynb
        |_ log_reg_2_optimized_cross_valid_all_data.ipynb
        |_ log_reg_2_optimized_cross_valid_all_data_feateng.ipynb
        |_ log_reg_2_optimized_cross_valid_all_data_feateng_filteryear.ipynb
        |_ log_reg_2_optimized_cross_valid_all_data_rm_dup.ipynb
        |_ log_reg_2_optimized_cross_valid_all_data_rm_out.ipynb
        |_ log_reg_2_optimized_cross_valid_report.ipynb
    |_ log_reg_2_optimized_model.ipynb
    |_ logistic_regression.py
    |_ preprocess.py
other_models/
    |_ LightGBM/
    |_ SVM/
requirements.txt
```