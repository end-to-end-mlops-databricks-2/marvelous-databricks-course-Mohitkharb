import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def get_power_consumption_of_tetouan_city() -> tuple[pd.DataFrame, pd.DataFrame]:
    from ucimlrepo import fetch_ucirepo
    # fetch dataset
    power_consumption_of_tetouan_city = fetch_ucirepo(id=849)

    # data (as pandas dataframes)
    X = power_consumption_of_tetouan_city.data.features
    y = power_consumption_of_tetouan_city.data.targets
    return X, y




def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the features of the power consumption data.
    """
    # 1. Handle timestamp
    features['DateTime'] = pd.to_datetime(features['DateTime'])
    features['Hour'] = features['DateTime'].dt.hour
    features['Day'] = features['DateTime'].dt.day
    features['Month'] = features['DateTime'].dt.month
    features['DayOfWeek'] = features['DateTime'].dt.dayofweek
    
    # Add seasonal features using cyclic encoding
    # Hour of day - 24 hour cycle
    features['Hour_sin'] = np.sin(2 * np.pi * features['Hour']/24)
    features['Hour_cos'] = np.cos(2 * np.pi * features['Hour']/24)
    
    # Day of week - 7 day cycle 
    features['DayOfWeek_sin'] = np.sin(2 * np.pi * features['DayOfWeek']/7)
    features['DayOfWeek_cos'] = np.cos(2 * np.pi * features['DayOfWeek']/7)
    
    # Month - 12 month cycle
    features['Month_sin'] = np.sin(2 * np.pi * features['Month']/12)
    features['Month_cos'] = np.cos(2 * np.pi * features['Month']/12)    
    
    # Drop original DateTime column and non-cyclic time features
    features = features.drop(['DateTime', 'Hour', 'Day', 'Month', 'DayOfWeek'], axis=1)

    # 2. Handle missing values if any
    features = features.fillna(features.mean())

    # 3. Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    
    return features_scaled, scaler
    


def preprocess_power_consumption_data(features: pd.DataFrame, targets: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    
    # 1. Preprocess features
    features_scaled, scaler = preprocess_features(features)
    
    
    # 2. Handle missing values in targets
    targets = targets.fillna(targets.mean())
    
    
    # 4. Split data for each zone
    # Assuming y has columns: ['Zone1', 'Zone2', 'Zone3']
    train_data = {}
    test_data = {}
    
    for zone in targets.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, 
            targets[zone],
            test_size=test_size,
            random_state=random_state
        )
        train_data[zone.replace(' ', '_')] = (X_train, y_train)
        test_data[zone.replace(' ', '_')] = (X_test, y_test)
    
    return train_data, test_data, scaler

def preprocess_tetuan_power_consumption_data() -> tuple:
    features, targets = get_power_consumption_of_tetouan_city()
    return preprocess_power_consumption_data(features, targets)