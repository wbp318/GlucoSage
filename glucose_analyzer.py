import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import os

def generate_sample_data(num_days=30):
    np.random.seed(42)  # for reproducibility
    dates = pd.date_range(start='2023-01-01', periods=num_days * 24, freq='H')
    glucose_levels = np.random.normal(loc=120, scale=30, size=len(dates))
    glucose_levels = np.clip(glucose_levels, 70, 200)  # Limit to realistic range
    
    df = pd.DataFrame({
        'timestamp': dates,
        'glucose_level': glucose_levels,
        'insulin_dose': np.random.uniform(0, 10, size=len(dates)),
        'carb_intake': np.random.uniform(0, 100, size=len(dates))
    })
    
    return df

class GlucoseAnalyzer:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            print(f"Data file not found. Generating sample data at {data_path}")
            df = generate_sample_data()
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
        
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.daily_avg = self.calculate_daily_average()

    def calculate_daily_average(self):
        return self.df.resample('D', on='timestamp')['glucose_level'].mean()

    def plot_daily_average(self, save_path):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=self.daily_avg.index, y=self.daily_avg.values)
        plt.title('Daily Average Glucose Levels')
        plt.xlabel('Date')
        plt.ylabel('Glucose Level (mg/dL)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def calculate_statistics(self):
        stats = self.df['glucose_level'].describe()
        stats['skewness'] = self.df['glucose_level'].skew()
        stats['kurtosis'] = self.df['glucose_level'].kurtosis()
        return stats

    def identify_extreme_events(self):
        hypo_events = self.df[self.df['glucose_level'] < 70]
        hyper_events = self.df[self.df['glucose_level'] > 180]
        return {
            'hypoglycemic_events': len(hypo_events),
            'hyperglycemic_events': len(hyper_events)
        }

    def perform_time_series_analysis(self):
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly_avg = self.df.groupby('hour')['glucose_level'].mean()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
        plt.title('Average Glucose Levels by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Glucose Level (mg/dL)')
        plt.tight_layout()
        plt.savefig('hourly_glucose_pattern.png')
        plt.close()

    def perform_clustering_analysis(self, n_clusters=3):
        X = self.df[['glucose_level', 'hour']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['cluster'] = kmeans.fit_predict(X_scaled)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.df, x='hour', y='glucose_level', hue='cluster', palette='viridis')
        plt.title('Glucose Level Clusters')
        plt.xlabel('Hour of Day')
        plt.ylabel('Glucose Level (mg/dL)')
        plt.tight_layout()
        plt.savefig('glucose_clusters.png')
        plt.close()

    def generate_report(self):
        self.plot_daily_average('daily_avg_glucose.png')
        stats = self.calculate_statistics()
        extreme_events = self.identify_extreme_events()
        self.perform_time_series_analysis()
        self.perform_clustering_analysis()

        report = f"""
        Glucose Analysis Report
        
        Basic Statistics:
        {stats}
        
        Extreme Events:
        {extreme_events}
        
        Time Series and Clustering Analysis:
        - Daily average glucose levels plot saved as 'daily_avg_glucose.png'
        - Hourly glucose pattern plot saved as 'hourly_glucose_pattern.png'
        - Glucose level clusters plot saved as 'glucose_clusters.png'
        """

        return report

# Usage
if __name__ == "__main__":
    analyzer = GlucoseAnalyzer('data/glucose_data.csv')
    report = analyzer.generate_report()
    print(report)