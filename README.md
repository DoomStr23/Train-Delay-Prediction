# Train Delay Prediction

This is a student project focused on predicting train delays using real-time timetable data from Deutsche Bahn. The goal is to build a complete pipeline that:
1. Collects the train data,
2. Cleans and processes the data,
3. Trains a machine learning model using engineered features,
4. Evaluates and saves the model for future use.

---

## Project Files

- `data_collection.py`  
  Collects real-time train arrival and departure data from Cologne Hauptbahnhof (Köln Hbf) using Deutsche Bahn’s API. The results are saved in `train_data_log.csv`.

- `clean_data.py`
  Cleans and filters the collected data, handling missing values and preparing the data for modeling. The cleaned output is saved as `clean_train_data.csv`.

- `random-forest_model_train.py`
  Trains a **Random Forest Regressor** on the cleaned dataset. Includes feature engineering, preprocessing with pipelines, hyperparameter tuning using GridSearchCV, model evaluation, and saving the final trained model.

---

## Outputs

- `train_data_log.csv`  
  Raw collected train data from the API.

- `clean_train_data.csv`  
  Cleaned dataset ready for training.

- `train_delay_model.joblib`  
  Final trained model saved using Joblib.

- `prediction_performance.png`  
  Scatter plot showing actual vs. predicted delay values.

- `error_distribution.png`  
  Histogram of prediction errors to analyze model accuracy.

---

## Model Performance

The model is evaluated using:
- **MAE** – Mean Absolute Error (average delay error)
- **RMSE** – Root Mean Squared Error (penalizes large errors)
- **R² Score** – Indicates how well the model explains the delay

All metrics are printed after training.

---

## Requirements

Install the necessary libraries with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib



