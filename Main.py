# Machine learning

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     StratifiedKFold, cross_validate,
                                     RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, make_scorer, 
                             accuracy_score, precision_score, recall_score, f1_score)

# Load dataset
df = pd.read_csv("C:\\Users\\kenye\\OneDrive\\SUSS\\ANL588 Project\\Applied Project\\CD Content Submission\\data\\GamingBehaviorData.csv")


###############################################
'''
Datatypes Alignment
'''

# Convert columns to categorial data types

Categorical_columns = ['Gender','Location','InGamePurchases',
                      'GameGenre','GameDifficulty','EngagementLevel']

for column in Categorical_columns:
    df[column] = df[column].astype('category')


# Define logical orders for GameDifficulty and EngagementLevel
GameDifficulty_order = ['Easy','Medium','Hard']
EngagementLevel_order = ["Low","Medium","High"]

# Convert GameDifficulty & EngagementLevel to categorial with specified order
df['GameDifficulty'] = pd.Categorical(df['GameDifficulty'], categories=GameDifficulty_order, ordered=True)
df['EngagementLevel'] = pd.Categorical(df['EngagementLevel'], categories=EngagementLevel_order, ordered=True)

# Set up numerical and categorical
numerical_fields = ['Age','PlayTimeHours','SessionsPerWeek',
                    'AvgSessionDurationMinutes','PlayerLevel','AchievementsUnlocked']
categorical_fields = ['Gender','Location','InGamePurchases','GameGenre','GameDifficulty','EngagementLevel']


###############################################

'''
Data pre-processing
Steps: (1)split data into training, validation and test sets, 
       (2)scaling features, 
       (3)encode categorial data
'''

# Drop irrelevant columns & split the data into features and target variable
X = df.drop(columns=['PlayerID', 'EngagementLevel'])
y = df['EngagementLevel']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the test-validation set (20%) into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of the splits to verify
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# Encode categorical and numerical variables
categorical_features = X.select_dtypes(include=['category']).columns
numerical_features = X.select_dtypes(include=[np.number]).columns

# Create preprocessor with specified categories
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)],
        remainder='passthrough')


###############################################
'''
Feature Importance Analysis
Decision Tree, Random Forest, Logistic Regression, Support Vector Machine, and Gradient Boosting
'''

# Define pipelines in a single dictionary
pipelines = {
    'dt_pipeline'  : Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())]),
    'rf_pipeline'  : Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())]),
    'lr_pipeline'  : Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())]),
    'svm_pipeline' : Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(kernel='linear'))]),
    'gb_pipeline'  : Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier())])
}
# Fit each pipeline and extract feature importances
for name, pipeline in pipelines.items():
    print(f"Fitting {name}...")
    
    # Fit the pipeline with training data
    pipeline.fit(X_train, y_train)
    
    # Get feature importances
    if name in ['dt_pipeline', 'rf_pipeline', 'gb_pipeline']:
        importances = pipeline.named_steps['classifier'].feature_importances_
    elif name == 'lr_pipeline':
        importances = abs(pipeline.named_steps['classifier'].coef_[0])  # Use absolute values for LR
    elif name == 'svm_pipeline':
        importances = abs(pipeline.named_steps['classifier'].coef_[0])  # Use absolute values for SVM
    
    # Get transformed feature names from OneHotEncoder
    ohe_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)

    # Get transformed feature names
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Combine numerical and categorical feature names
    transformed_columns = numerical_features + ohe_feature_names.tolist()

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({'Feature': transformed_columns, 'Importance': importances})

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Display the most important features
    print(f"Key Variables Considered by {name}:")
    print(importance_df)

    # Plotting the feature importances
    plt.figure(figsize=(22, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title(f'Feature Importances from {name}')
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance on top

plt.show()



###############################################
'''
This section is for the kNN model and it determines the ideal number of neighbors.
The ideal number of neighbours will be used to train the model and also for features importance analysis.
'''

# Define a range of n_neighbors values to test
neighbors_range = range(1, 31)  # Testing from 1 to 30
mean_scores = []

# Loop through each value of n_neighbors
for n in neighbors_range:
    # Create a k-NN pipeline with the current n_neighbors
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=n))
    ])
    
    # Perform cross-validation and calculate mean accuracy
    scores = cross_val_score(knn_pipeline, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold CV
    mean_scores.append(scores.mean())

# Find the optimal number of neighbors
optimal_k = neighbors_range[np.argmax(mean_scores)]
print(f"Optimal number of neighbors: {optimal_k}")

'''
Feature importance analysis for kNN
'''
# Pipeline for kNN model
n_neighbors = 28  # number derived from 'determine ideal kNN (above)'
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors))
])

# Train the kNN model
knn_pipeline.fit(X_train,y_train)

# Make predictions with kNN
y_pred_knn = knn_pipeline.predict(X_val)

# Evalate the kNN model
print(f"kNN Classification Report (n_nieghbors={n_neighbors}):\n",
      classification_report(y_val, y_pred_knn))
print("kNN Confusion Matrix:\n", confusion_matrix(y_val, y_pred_knn))

# Note: kNN does not provide feature importance.
# Calculate permutation importance
result = permutation_importance(knn_pipeline, X_train, y_train, n_repeats=30, random_state=42)

# Get transformed feature names for tabular display
ohe_feature_names = knn_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)

# Get transformed feature names
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Combine numerical and categorical feature names
transformed_columns = numerical_features + ohe_feature_names.tolist()

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': transformed_columns,  # Use all transformed columns
    'Importance': np.concatenate([result.importances_mean, np.zeros(len(transformed_columns) - len(result.importances_mean))])  # Fill remaining importances with zeros
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances in tabular format
print("Permutation Importance of Features for kNN model:")
print(importance_df.reset_index(drop=True))

###############################################
'''
Compare different classification models
'''

# Define models to compare
models = [
    ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42)),
    ('RandomForest', RandomForestClassifier(max_features='sqrt', random_state=42)),
    ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
    ('kNN', KNeighborsClassifier(n_neighbors=28)),
    ('SVC', SVC(random_state=42)),
    ('GradientBoosting', GradientBoostingClassifier(random_state=42))
]

# Define the cross-validation strategy
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# Define the scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

results = {}
for name, model in models:
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    start_time = time.time()  # Start timing
    
    cv_results = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
    
    end_time = time.time()  # End timing
    runtime = end_time - start_time  # Calculate runtime
    
    results[name] = cv_results
    results[name]['runtime'] = runtime  # Store runtime

# Display cross-validation results along with runtime
for name, result in results.items():
    print(f"\n{name} Model Performance:")
    print(f"Accuracy: {result['test_accuracy'].mean():.2f} +/- {result['test_accuracy'].std():.2f}")
    print(f"Precision: {result['test_precision'].mean():.2f} +/- {result['test_precision'].std():.2f}")
    print(f"Recall: {result['test_recall'].mean():.2f} +/- {result['test_recall'].std():.2f}")
    print(f"F1 Score: {result['test_f1'].mean():.2f} +/- {result['test_f1'].std():.2f}")
    print(f"Runtime: {result['runtime']:.2f} seconds")


###############################################
'''
Hyperparameter Tuning using Random Search on Random Forest and Gradient Boosting
'''

# Hyperparameter distribution for Random Forest
rf_param_dist = {
    'classifier__n_estimators': np.arange(100, 1001, 100),
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Hyperparameter distribution for Gradient Boosting
gb_param_dist = {
    'classifier__n_estimators': np.arange(100, 1001, 100),
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Create pipelines for both models
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# Perform Random Search for Random Forest
rf_random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=kfold,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

rf_random_search.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_random_search.best_params_)
print("Best score for Random Forest:", rf_random_search.best_score_)

# Perform Random Search for Gradient Boosting
gb_random_search = RandomizedSearchCV(
    estimator=gb_pipeline,
    param_distributions=gb_param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=kfold,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

gb_random_search.fit(X_train, y_train)
print("Best parameters for Gradient Boosting:", gb_random_search.best_params_)
print("Best score for Gradient Boosting:", gb_random_search.best_score_)


###############################################
'''
Model Training on Random Forest and Gradient Boosting
'''

# Best hyperparameters obtained from tuning
best_rf_params = {
    'n_estimators': 1000,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': None
}

best_gb_params = {
    'n_estimators': 900,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'max_depth': 7,
    'learning_rate': 0.01
}

# Create and train the Random Forest model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**best_rf_params, random_state=42))
])

# Start measuring runtime
start_time = time.time()

# Train the Random Forest model
rf_pipeline.fit(X_train, y_train)

# End measuring runtime
end_time = time.time()
runtime = end_time - start_time

print(f"Random Forest model trained in {runtime:.2f} seconds.")
print("Random Forest model trained with best parameters.")


# Create and train the Gradient Boosting model
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(**best_gb_params, random_state=42))
])

# Start measuring runtime
start_time = time.time()

# Train the Gradient Boosting model
gb_pipeline.fit(X_train, y_train)

# End measuring runtime
end_time = time.time()
runtime = end_time - start_time

print(f"Gradient Boosting model trained in {runtime:.2f} seconds.")
print("Gradient Boosting model trained with best parameters.")

###############################################
'''
Model Evaluation
'''

# Evaluate the Random Forest model on the test set
rf_y_pred = rf_pipeline.predict(X_test)

# Calculate performance metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)
rf_confusion = confusion_matrix(y_test, rf_y_pred)

print("Random Forest Model Evaluation:")
print("Accuracy:", rf_accuracy)
print("\nClassification Report:\n", rf_report)
print("Confusion Matrix:\n", rf_confusion)


# Evaluate the Gradient Boosting model on the test set
gb_y_pred = gb_pipeline.predict(X_test)

# Calculate performance metrics for Gradient Boosting
gb_accuracy = accuracy_score(y_test, gb_y_pred)
gb_report = classification_report(y_test, gb_y_pred)
gb_confusion = confusion_matrix(y_test, gb_y_pred)

print("\nGradient Boosting Model Evaluation:")
print("Accuracy:", gb_accuracy)
print("\nClassification Report:\n", gb_report)
print("Confusion Matrix:\n", gb_confusion)

# Based on evaluation result, Gradient Boosting is the best model

# END OF CODE



