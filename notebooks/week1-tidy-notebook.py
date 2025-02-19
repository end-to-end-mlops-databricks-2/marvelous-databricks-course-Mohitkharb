# Databricks notebook source

# COMMAND ----------
!pip install power_consumption

# COMMAND ----------
from power_consumption.get_data import preprocess_power_consumption_data

train_data, test_data, scaler = preprocess_power_consumption_data()

# COMMAND ----------
print("First few rows of train_data:")
for key in train_data.keys():
    print(f"\n{key}:")

print("\nFirst few rows of test_data:")
for key in test_data.keys():
    print(f"\n{key}:")

# COMMAND ----------
train_data['Zone_1_Power_Consumption'][:5]

# COMMAND ----------
test_data['Zone_1_Power_Consumption'][:5]



# COMMAND ----------
for zone, (X_train, y_train) in train_data.items():
    X_test, y_test = test_data[zone]
    