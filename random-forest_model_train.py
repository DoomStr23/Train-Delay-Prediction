import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# STEP 1: Load your cleaned dataset
df = pd.read_csv("clean_train_data.csv")

# STEP 2: Feature Engineering
# Convert planned_time to hour and minute features
df['planned_hour'] = df['planned_time'].apply(lambda x: int(x.split(':')[0]))
df['planned_minute'] = df['planned_time'].apply(lambda x: int(x.split(':')[1]))

# Calculate time of day periods
df['time_period'] = pd.cut(
    df['planned_hour'],
    bins=[0, 6, 12, 18, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening']
)

# Create a feature for rush hours (7-9 AM and 4-7 PM)
df['is_rush_hour'] = ((df['planned_hour'] >= 7) & (df['planned_hour'] <= 9)) | \
                     ((df['planned_hour'] >= 16) & (df['planned_hour'] <= 19))

# Extract day of week as numeric (0=Monday, 6=Sunday)
df['day_of_week_num'] = pd.to_datetime(df['planned_date']).dt.dayofweek

# Create is_weekend feature
df['is_weekend'] = df['day_of_week_num'].apply(lambda x: 1 if x >= 5 else 0)

# STEP 3: Define features (X) and target (y)
# Categorical features
cat_features = ['station_eva', 'direction', 'train_type', 'status', 'time_period', 'day_of_week']
# Numerical features
num_features = ['planned_hour', 'planned_minute', 'is_rush_hour', 'is_weekend', 'day_of_week_num', 'train_number']

# Combine features
features = cat_features + num_features
X = df[features].copy()
y = df['delay_minutes']

# STEP 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Create preprocessing pipeline
# Preprocessing for categorical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing for numerical features
numerical_transformer = StandardScaler()

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# STEP 6: Create a preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# STEP 7: Define hyperparameter grid
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10]
}

# STEP 8: Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# STEP 9: Make predictions
y_pred = best_model.predict(X_test)

# STEP 10: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
# Calculate RMSE using the proper approach (sqrt of MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"  - R² Score: {r2:.2f}")
print(f"\nBest Parameters: {grid_search.best_params_}")

# STEP 11: Save the model
joblib.dump(best_model, 'train_delay_model.joblib')
print("\n Model saved as 'train_delay_model.joblib'")

# STEP 12: Visualize Results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Predicted Delay (minutes)')
plt.title('Actual vs Predicted Delay')
plt.tight_layout()
plt.savefig('prediction_performance.png')
plt.show()

# STEP 13: Analyze Error Distribution
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel('Prediction Error (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.tight_layout()
plt.savefig('error_distribution.png')
plt.show()

