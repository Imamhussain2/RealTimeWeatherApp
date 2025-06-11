import pandas as pd
import numpy as np
import random
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

# -----------------------------------------------
# Step 1: Synthetic Data Generation
# -----------------------------------------------
def generate_synthetic_weather_data():
    cities =  [
                'ahmedabad', 'assam', 'bengaluru', 'chennai', 'delhi',
                'hyderabad', 'jaipur', 'kolkata', 'lucknow', 'mumbai',
                'panaji', 'pune', 'shimla', 'srinagar', 'thiruvananthapuram'
    ]
    weather_conditions = [
                "clear sky", "few clouds", "scattered clouds", "broken clouds",
                "shower rain", "rain", "thunderstorm", "snow", "mist"
            ]
    data = []

    for city in cities:
        for condition in weather_conditions:
            for _ in range(1500):
                temp = round(random.uniform(20, 40), 2)
                pressure = random.randint(990, 1025)
                humidity = random.randint(40, 90)
                wind_speed = round(random.uniform(0, 10), 2)
                wind_gust = wind_speed + round(random.uniform(0, 5), 2)
                clouds = random.randint(0, 100)
                wind_deg = random.uniform(0, 360)
                hour = random.randint(0, 23)
                is_day = int(6 <= hour <= 18)

                data.append({
                    'city': city,
                    'temperature_celsius': temp,
                    'pressure_hpa': pressure,
                    'humidity_percent': humidity,
                    'wind.speed_m_s': wind_speed,
                    'wind.gust_m_s': wind_gust,
                    'clouds_percent': clouds,
                    'wind_dir_sin': np.sin(np.deg2rad(wind_deg)),
                    'wind_dir_cos': np.cos(np.deg2rad(wind_deg)),
                    'hour_of_day': hour,
                    'is_daytime': is_day,
                    'weather.main': condition
                })

    return pd.DataFrame(data)

# Generate the dataset
df = generate_synthetic_weather_data()

# -----------------------------------------------
# Step 2: Preprocessing and Model Pipeline
# -----------------------------------------------
features = [
    'city', 'temperature_celsius', 'pressure_hpa', 'humidity_percent',
    'wind.speed_m_s', 'wind.gust_m_s', 'clouds_percent',
    'wind_dir_sin', 'wind_dir_cos', 'hour_of_day', 'is_daytime'
]
X = df[features]
y = df['weather.main']

numeric_features = [
    'temperature_celsius', 'pressure_hpa', 'humidity_percent',
    'wind.speed_m_s', 'wind.gust_m_s', 'clouds_percent',
    'wind_dir_sin', 'wind_dir_cos', 'hour_of_day', 'is_daytime'
]
categorical_features = ['city']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,
    scoring='f1_macro',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Train the model
search.fit(X_train, y_train)

print("✅ Best parameters:", search.best_params_)
print(f"✅ Best CV f1_macro score: {search.best_score_:.4f}")

# Evaluate on test set
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print("\n🧪 Test Set Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(best_model, "weather_model_balanced_tuned.pkl")
print("📦 weather_model_balanced_tuned.pkl saved successfully.")
