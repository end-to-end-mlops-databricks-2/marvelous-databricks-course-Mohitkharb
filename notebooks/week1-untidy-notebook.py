# Databricks notebook source

# MAGIC %md
# MAGIC # Tidy Data
# MAGIC 
# MAGIC 


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load install packages

!pip install ucimlrepo


# COMMAND ----------

from ucimlrepo import fetch_ucirepo


power_consumption_of_tetouan_city = fetch_ucirepo(id=849)

X = power_consumption_of_tetouan_city.data.features
Y = power_consumption_of_tetouan_city.data.targets

X.head()



# COMMAND ----------
Y.head()

# COMMAND ----------
print(X.shape, Y.shape)




# COMMAND ----------
print(power_consumption_of_tetouan_city.data)
# COMMAND ----------
