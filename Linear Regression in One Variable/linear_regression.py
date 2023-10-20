import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# # Predicting House Prices based on Area
# df = pd.read_csv("house_prices.csv")

# plt.xlabel("Area(sq.ft.)")
# plt.ylabel("Price($)")
# plt.scatter(df.Area, df.Price)

# reg = linear_model.LinearRegression()
# reg.fit(df[['Area']], df.Price)

# print(reg.predict([[3300]]))

# Predicting Canada PCI

df = pd.read_csv("canada_pci.csv")
print(df[['year']])
reg = linear_model.LinearRegression()
reg.fit(df[['year']],df[['pci']])
print(reg.predict([[2021]]))