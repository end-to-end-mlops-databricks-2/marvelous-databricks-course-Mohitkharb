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

def preprocess_power_consumption_data() -> tuple:
    # Get raw data
    X, y = get_power_consumption_of_tetouan_city()
    
    # 1. Handle timestamp
    X['DateTime'] = pd.to_datetime(X['DateTime'])
    X['Hour'] = X['DateTime'].dt.hour
    X['Day'] = X['DateTime'].dt.day
    X['Month'] = X['DateTime'].dt.month
    X['DayOfWeek'] = X['DateTime'].dt.dayofweek
    
    # Add seasonal features using cyclic encoding
    # Hour of day - 24 hour cycle
    X['Hour_sin'] = np.sin(2 * np.pi * X['Hour']/24)
    X['Hour_cos'] = np.cos(2 * np.pi * X['Hour']/24)
    
    # Day of week - 7 day cycle
    X['DayOfWeek_sin'] = np.sin(2 * np.pi * X['DayOfWeek']/7)
    X['DayOfWeek_cos'] = np.cos(2 * np.pi * X['DayOfWeek']/7)
    
    # Month - 12 month cycle
    X['Month_sin'] = np.sin(2 * np.pi * X['Month']/12)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month']/12)
    
    # Drop original DateTime column and non-cyclic time features
    X = X.drop(['DateTime', 'Hour', 'Day', 'Month', 'DayOfWeek'], axis=1)
    
    
    # 2. Handle missing values if any
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # 3. Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 4. Split data for each zone
    # Assuming y has columns: ['Zone1', 'Zone2', 'Zone3']
    train_data = {}
    test_data = {}
    
    for zone in y.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, 
            y[zone],
            test_size=0.2,
            random_state=42
        )
        train_data[zone.replace(' ', '_')] = (X_train, y_train)
        test_data[zone.replace(' ', '_')] = (X_test, y_test)
    
    return train_data, test_data, scaler

