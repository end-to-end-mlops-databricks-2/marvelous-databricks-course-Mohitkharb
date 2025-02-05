
from ucimlrepo import fetch_ucirepo


def get_data(num_records: int = 1000):
    # fetch dataset
    power_consumption_of_tetouan_city = fetch_ucirepo(id=849)
    
    # data (as pandas dataframes)
    X = power_consumption_of_tetouan_city.data.features
    y = power_consumption_of_tetouan_city.data.targets
    return X, y