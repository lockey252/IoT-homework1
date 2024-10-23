import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CRISP-DM: Business Understanding: We are trying to predict a linear relationship between x and y
st.title("Interactive Simple Linear Regression App")

# Data Understanding: User can modify slope (a), intercept (b), noise level, and number of data points
st.sidebar.header("User Inputs")
a = st.sidebar.slider('Slope (a)', min_value=-10.0, max_value=10.0, value=2.0)
b = st.sidebar.slider('Intercept (b)', min_value=-10.0, max_value=10.0, value=1.0)
noise_level = st.sidebar.slider('Noise Level', min_value=0.0, max_value=5.0, value=1.0)
num_points = st.sidebar.slider('Number of Points', min_value=10, max_value=1000, value=100)

# Data Preparation: Generating synthetic data
X = np.random.rand(num_points, 1) * 10  # Independent variable
noise = np.random.randn(num_points, 1) * noise_level
y = a * X + b + noise  # Dependent variable with noise

# Modeling: Perform simple linear regression
model = LinearRegression()
model.fit(X, y)

# Evaluation: Plotting the results
y_pred = model.predict(X)

# Displaying the original data points and the predicted line
fig, ax = plt.subplots()
ax.scatter(X, y, label='Data Points')
ax.plot(X, y_pred, color='red', label='Fitted Line')
ax.set_title(f'Linear Regression: y = {a}x + {b}')
ax.legend()

st.pyplot(fig)

# Evaluation: Show the slope and intercept of the fitted line
st.write(f"Fitted line equation: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")

# Allow the user to download the dataset
import pandas as pd
df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})
st.download_button("Download Dataset", df.to_csv(index=False), file_name="dataset.csv")
