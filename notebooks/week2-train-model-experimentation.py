# Databricks notebook source

# COMMAND ----------
pip install power_consumption





# COMMAND ----------
from power_consumption.config import ProjectConfig

project_config = ProjectConfig.from_yaml("../config/tetuan_power.yaml")
print(project_config)


# COMMAND ----------
from power_consumption.get_data import preprocess_tetuan_power_consumption_data


train_data, test_data, scaler = preprocess_tetuan_power_consumption_data()


# COMMAND ----------
from power_consumption.train_model import train_and_evaluate

models, results = train_and_evaluate(train_data, test_data, project_config)


# COMMAND ----------
print(results)



# COMMAND ----------
for zone, (X_test, y_test) in test_data.items():
    print(f"\nZone: {zone}")
    print(f"Average power consumption: {y_test.mean():.2f} kW")
    print(f"Min power consumption: {y_test.min():.2f} kW")
    print(f"Max power consumption: {y_test.max():.2f} kW")
    
    # Calculate percentage error
    rmse = results[zone]['rmse']
    percentage_error = (rmse / y_test.mean()) * 100
    print(f"RMSE: {rmse:.2f} kW")
    print(f"Error percentage: {percentage_error:.2f}%")
# COMMAND ----------
