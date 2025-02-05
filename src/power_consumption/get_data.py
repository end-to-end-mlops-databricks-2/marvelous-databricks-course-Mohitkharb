
from ucimlrepo import fetch_ucirepo
import pandas as pd

def get_data(id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # fetch dataset
    power_consumption_of_tetouan_city = fetch_ucirepo(id=id)
    
    # data (as pandas dataframes)
    X = power_consumption_of_tetouan_city.data.features
    y = power_consumption_of_tetouan_city.data.targets
    return X, y