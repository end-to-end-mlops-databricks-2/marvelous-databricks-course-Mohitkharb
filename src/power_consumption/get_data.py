# Databricks notebook source

from ucimlrepo import fetch_ucirepo

# fetch dataset
power_consumption_of_tetouan_city = fetch_ucirepo(id=849)

# data (as pandas dataframes)
X = power_consumption_of_tetouan_city.data.features
y = power_consumption_of_tetouan_city.data.targets

# metadata
print(power_consumption_of_tetouan_city.metadata)

# variable information
print(power_consumption_of_tetouan_city.variables)
