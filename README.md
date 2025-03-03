# Linear Regression: Parameter Optimization using Linear Search and Gradient Descent

This project demonstrates how to estimate the best-fit parameters (`m1` and `m2`) for a simple linear regression model using **Linear Search** and **Gradient Descent**. The code was executed in **Google Colab**.

## 📌 Overview

We generate synthetic data based on a linear equation:



## 🧩 **Formulas Used**
## 1️⃣ Linear Transformation
\[
y = m_1x + m_2 + n(0,1)
\]

- \( y \) → Output value  
- \( m_1 \) → Slope  
- \( x \) → Input variable  
- \( m_2 \) → Intercept
- \(n(0,1)\) → Noise

---

## 2️⃣ Gradient Calculation
Measures the rate of change of \( f(x) \).

---

## 3️⃣ Loss Function
\[
MSE = frac 1/ N sum_i=1 to N (y - yi)^2
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

![Result](https://github.com/user-attachments/assets/89ead175-3302-465b-8f83-bbc1f767662f)

!
