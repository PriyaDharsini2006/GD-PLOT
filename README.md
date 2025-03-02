# Linear Regression: Parameter Optimization using Linear Search and Gradient Descent

This project demonstrates how to estimate the best-fit parameters (`m1` and `m2`) for a simple linear regression model using **Linear Search** and **Gradient Descent**. The code was executed in **Google Colab**.

## 📌 Overview

We generate synthetic data based on a linear equation:

\[
y = m1_{\text{true}} \times x + m2_{\text{true}} + \text{noise}
\]

Then, we apply:
- **Linear Search** to find the best `m1` minimizing Mean Squared Error (MSE).
- **Gradient Descent** to iteratively optimize `m1` and `m2`.

## 📊 Results

### 1️⃣ Linear Search: MSE vs. `m1`
This method evaluates different values of `m1` within a range and selects the one that results in the lowest MSE.
![Linear](https://github.com/user-attachments/assets/bfcc6cc3-1b83-4430-bf27-85ea4e4e5632)

### 2️⃣ Gradient Descent: MSE vs. `m1` 
Gradient Descent updates `m1` and `m2` iteratively based on computed gradients.
![Gradient](https://github.com/user-attachments/assets/f77d2cde-dc59-4681-966d-5680964d230a)


![Results](https://github.com/user-attachments/assets/a536f22c-1f54-4735-becc-50464e426a13)
