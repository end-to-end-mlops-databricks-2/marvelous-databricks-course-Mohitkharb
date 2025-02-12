# Databricks notebook source

# COMMAND ----------
!pip install power_consumption

# COMMAND ----------
from power_consumption.get_data import preprocess_power_consumption_data

train_data, test_data, scaler = preprocess_power_consumption_data()

# COMMAND ----------
train_data.keys()

# COMMAND ----------
train_data['Zone_1_Power_Consumption']

# COMMAND ----------



