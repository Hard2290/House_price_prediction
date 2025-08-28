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
- Creating 5 folds using "KFold" and mentioning the validation folds in a separate column "kfold".

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
- Extracted a repeating pattern of data out of the unique ticket details of all the passengers onboard in "Ticket" column and created a new categorical feature named "ticket_type".
- Different "ticket_type" values showing different "Survival" probabilities for passengers.
![Image](https://github.com/user-attachments/assets/1ac5c2b8-a2fc-4bfb-9314-920f8e6f1540)

- Similarly, using "Cabin" feature containing alpha-numeric cabin name alloted to passsengers, a categorical feature has been created and named "cabin_type".
- Different "cabin_type" values showing different "Survival" probabilities for passengers.
![Image](https://github.com/user-attachments/assets/69534fd7-4756-4bac-9b37-20eae8cc6fef)

- Using unique "Name" column, two informations have been extracted, family to which the passengers belongs to and their title and these informations presented under two new categorical features, "family" and "title" respectively.
- Different "title" values showing different "Survival" probabilities for passengers.
![Image](https://github.com/user-attachments/assets/e8a3bade-1abd-4766-9b28-827b2d43c0bb)

- Created two new features; "family_size", using "SibSp" and "Parch" columns, and "family_count", containing count of all the families using "family" feature.
- Created two new features; "Age_NA", showing positions of missing values in "Age" column, and "Age_mean", containing Mean Imputation of "Age" column as data was nearly symmetrical.

## Handling Outliers:
- Using boxplot, outliers detected in "Age", "Age_mean" and "Fare" features.
![Image](https://github.com/user-attachments/assets/2de31ad2-62b5-4ed8-963f-f90229aab96e)

![Image](https://github.com/user-attachments/assets/7fc4a3ed-0cb1-4697-92c1-c24e0da6517d)

![Image](https://github.com/user-attachments/assets/680c170a-83f0-4b38-8729-0d59816dd625)

- Data distribution of "Age" and "Age_mean" is nearly symmetrical, so to deal with outliers, their values have been bounded within 3σ wrt mean.
![Image](https://github.com/user-attachments/assets/38a660a0-15ff-485e-8ec6-9ee658936b04)
![Image](https://github.com/user-attachments/assets/066da803-b1b7-49c1-8220-5968aa0489b9)

- Since "Fare" has skewed data distribution, so its values have been bounded within 3IQR (Interquartile Range) from 75th and 25th percentile value.
![Image](https://github.com/user-attachments/assets/c7a68ddc-a20a-47a9-93b2-05e35a20eafe)

## Feature Encoding and Modeling:
- 4 types of Feature Encodings, "One-Hot Encoding", "Label Encoding", "Mean Encoding" and "Frequency Encoding" used.
- 5 types of classifiers, "Random Forest Classifier", "Gradient Boosting Classifier", "KNeighbors Classifier", "Support Vector Classifier" and "XGBoost Classifier" used.
- Using "Optuna" for "XGBoost" and manual "Grid Search" for rest of the classifiers, Hyperparameter tuning done  over 5 cross-validation folds for all the 20 Classifier-Encoding combinations to obtain the best performing model in terms of accuracy.
- Final Hyperparameters of the best performing models obtained after further careful tuning of parameters to remove overfitting or underfitting as indicated by the differences in accuracy scores of the validation and test dataset according to scores on Kaggle's Public Leaderboard.
![Image](https://github.com/user-attachments/assets/57068e2c-f97c-4c33-a304-37906fcfaa69)
![Image](https://github.com/user-attachments/assets/50ff5fac-b457-4e7c-ab62-8696be4b0f28)
![Image](https://github.com/user-attachments/assets/37467fcf-e705-4407-b8b5-7acb89c201b7)
![Image](https://github.com/user-attachments/assets/7310a9aa-8480-4b13-897d-86448fd48d2c)
![Image](https://github.com/user-attachments/assets/8a3f0a16-1622-449d-a2f2-54b1778316df)

## Final Submission:
- Below are the Top 5 best performing models out of 20 models.
![Image](https://github.com/user-attachments/assets/c24f6284-8e34-44e7-8bfa-eb4e318502d4)

- Final prediction obtained by taking mean of the top 3 best performing models and converting them to binary (0 & 1) by choosing 0.5 as threshold.
- An accuracy of 0.80382 has been achieved by the model on Kaggle's Public Leaderboard.
