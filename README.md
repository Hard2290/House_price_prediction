# Housing Prices Competition-Kaggle
- Kaggle competition link - https://www.kaggle.com/competitions/home-data-for-ml-course/overview
- This is a Regression ML problem.
- The dataset contains 80 features of 1460 houses along with the prices at which these houses were sold mentioned under "SalePrice", which is our target feature. 

## Lowest RMSE achieved on Kaggle Leaderboard - 0000000

## Problem Statement :
- Build a model to predict the sales price for each house in test dataset.
- Model's performance will be evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.

## Tools used :
- Python - Coding Language
- Pandas - For Data Processing
- Numpy - For Arrays and Computations
- Sklearn - For ML Models and Metrics
- Optuna - For Hyperparameter Tuning
- Matplotlib/Seaborn - For Data Visualization

## Creating Cross-validation Folds :
- Creating and using same cross-validation folds throughout to prevent any overfitting.
- Creating 5 folds using "KFold" and mentioning the validation folds in a separate column "kfold" having value from 0 - 4.

![Image](https://github.com/user-attachments/assets/9be1e413-7848-41df-b90c-733fd5c8a0ab)

## Handling Missing values and Outliers:
#### For Categorical Data
- Features with more than 80% of data missing have been removed from the dataset.
- Features with more than 40% missing data have been imputed with "missing" category.
- Mode Imputation done for rest of the features. 
#### For Numerical Data
- Median Imputation done for features having less than 0.5% data missing data.

![Image](https://github.com/user-attachments/assets/655cb31e-ba50-456d-a830-7e6cbd364c07)

- Median Imputation also done for features "GarageYrBlt" and "MasVnrArea" that have skewed distribution, i.e., has outliers.

![Image](https://github.com/user-attachments/assets/84116528-6847-4ea7-b5ad-6b6a0085621a)

- Feature "LotFrontage" is approx. normally distributed and it has a lot of outliers as shown by Boxplot below.

![Image](https://github.com/user-attachments/assets/01af1cc8-c1f5-42e4-ae8b-c1371efd40b0)

![Image](https://github.com/user-attachments/assets/1d4da5c3-9a9e-4a07-89a5-c3012b0474a7)

- Values of "LotFrontage" has been bounded within 2σ to reduce outliers effect and a new Mean-Imputed feature "LotFrontage_mean" created.
- Predicted missing values for "LotFrontage" feature using mean of predictions from two best performing models out of "Random Forest Regression", "KNeighbors Regression" and "Gradient Boosting Regression" models trained over 10 validation folds while using "One-Hot Encoding" and "CatBoost Encoding" to encode low cardinal and high cardinal categorical features respectively.

![Image](https://github.com/user-attachments/assets/9c4c6ec5-1aaf-402f-9a6c-6b22bd4711c4)

- Flag features have been created for each of "MasVnrArea", "GarageYrBlt" and "LotFrontage" indicating position of missing values in them by "1" for missing and "0" for not missing.

## Feature Engineering:
- Features having more than 80% data "0" and is logically related to categorical features that have more than 80% data missing, are removed.
- Feature Engineering mostly done with features having more than 70% data "0" or is related to them.
- Features created to account for number of bathrooms and number of porch features houses have.
- Flag/Binary features created to indicate presence of "OpenPorch", "WoodDeck" and "Basement Bathrooms" in the houses with "1" indicating presence and "0" for absence.
- Features showing how latest remodeling has been done and the age of houses with respect to the sale year, have been created.
- A feature for average basement finished area per quality of each of First finish and Second finish and a feature for average total area per quality of each of Basement and Garage have been created.
- Feature indicating whether the living area has low quality finish or not has also been created.
- Created a Merged One-Hot Encoding of "Condition1" and "Condition2" features as they were related because "Condition2" occur only after "Condition1" has value "Norm" otherwise "Condition2" will have mostly "Norm" as value.
- Created count features denoting number of positive locations (Good for house value/price) and number of negative locations nearby houses.

## Handling Rare Categories:
- No features were found having any rare category in test dataset that is not present in training dataset.

![Image](https://github.com/user-attachments/assets/e59e3cc6-780a-4a49-a4c6-02803eefdfe4)

## Permutation Importance:
- Feature Importances obtained using Permutation Importance for each of the 4 encoding pairs mentioned in below section.
- Almost all of the top 15 important features were used during feature engineering.

![Image](https://github.com/user-attachments/assets/382376f8-0728-425f-b29f-569fe2121e4d)

- Out of the 20 least important features, it is unsafe to remove features either created out of encoding or during feature engineering as they might have good correlation with other similar encoded/created features. Only "MoSold" and "Id" were the only base features present in this list. Also, "MoSold" has the 4th lowest feature importance, having weight of -6.089±45.594. It has very high standard deviation and negative mean, so can remove it.

## Feature Encoding and Modeling:
- 4 types of Feature Encoding pairs used : 
  + #### One-Hot Encoding + CatBoost Encoding
  + #### One-Hot Encoding + Frequency Encoding
  + #### Categorical + CatBoost Encoding
  + #### Categorical + Frequency Encoding
  For Low Cardinal Categorical Features -> One-Hot Encoding and Categorical.
  
  For High Cardinal Categorical Features -> CatBoost Encoding and Frequency Encoding 
- 3 types of Regression Models used : 
  + #### Random Forest Regression
  + #### Gradient Boosting Regression
  + #### XGBoost Regression
- Using "Optuna" for "XGBoost" and manual "Grid Search" for rest of the regression models, Hyperparameter tuning done over 5 cross-validation folds for all the 12 Regressor-Encoding combinations to obtain the best performing models.
- Final Hyperparameters of the best performing models obtained after further careful tuning of parameters to reduce leakage and improve generalization which reduce errors further on Kaggle's Public Leaderboard.
![Image](https://github.com/user-attachments/assets/57068e2c-f97c-4c33-a304-37906fcfaa69)
![Image](https://github.com/user-attachments/assets/50ff5fac-b457-4e7c-ab62-8696be4b0f28)
![Image](https://github.com/user-attachments/assets/37467fcf-e705-4407-b8b5-7acb89c201b7)
![Image](https://github.com/user-attachments/assets/7310a9aa-8480-4b13-897d-86448fd48d2c)
![Image](https://github.com/user-attachments/assets/8a3f0a16-1622-449d-a2f2-54b1778316df)

## Final Submission:
- Below are the Top 5 best performing models out of 12 models.
![Image](https://github.com/user-attachments/assets/c24f6284-8e34-44e7-8bfa-eb4e318502d4)

- Final prediction of "SalePrice" obtained by taking mean of predictions from the top 3 best performing models.
- Lowest RMSE of  has been achieved on Kaggle's Public Leaderboard.
