import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the CSV data
data = pd.read_csv('Data/0-calibration_fts-accel.csv')
# Plot the data
data.plot()
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Data Plot')
plt.show()