import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class GlucosePredictor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.model = None
        self.scaler = None

    def preprocess_data(self):
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        
        # Feature engineering
        self.df['glucose_lag1'] = self.df['glucose_level'].shift(1)
        self.df['glucose_lag2'] = self.df['glucose_level'].shift(2)
        self.df['glucose_diff'] = self.df['glucose_level'].diff()
        self.df['insulin_lag1'] = self.df['insulin_dose'].shift(1)
        
        self.df = self.df.dropna()

        X = self.df[['hour', 'day_of_week', 'insulin_dose', 'carb_intake', 'glucose_lag1', 'glucose_lag2', 'glucose_diff', 'insulin_lag1']]
        y = self.df['glucose_level']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ])

        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        
        # Save the model
        joblib.dump(self.model, 'glucose_predictor_model.joblib')

        return self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Plotting actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Glucose Level')
        plt.ylabel('Predicted Glucose Level')
        plt.title('Actual vs Predicted Glucose Levels')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted_glucose.png')
        plt.close()

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def predict_glucose(self, input_data):
        return self.model.predict(input_data)

    def feature_importance(self):
        feature_importance = pd.DataFrame({
            'feature': self.model.named_steps['model'].feature_names_in_,
            'importance': self.model.named_steps['model'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        return feature_importance

# Usage
if __name__ == "__main__":
    predictor = GlucosePredictor('data/diabetes_data.csv')
    evaluation_metrics = predictor.train_model()
    print("Model Evaluation Metrics:", evaluation_metrics)
    print("\nFeature Importance:")
    print(predictor.feature_importance())