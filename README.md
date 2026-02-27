# Medical Cost Prediction — NumPy Linear Regression

A multiple linear regression model built **from scratch with NumPy** to predict individual medical insurance charges.  
This is a learning project that implements gradient descent, z-score normalisation, and a cost function without using any ML framework.

---

## Dataset

The project uses the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (`insurance.csv`), which contains **1,338 records** with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `age` | integer | Age of the primary beneficiary |
| `sex` | string | Insurance contractor gender (`male` / `female`) |
| `bmi` | float | Body mass index |
| `children` | integer | Number of children covered by insurance |
| `smoker` | string | Whether the beneficiary smokes (`yes` / `no`) |
| `region` | string | Residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`) |
| `charges` | float | Individual medical costs billed by health insurance (target variable) |

---

## Project Overview

### 1. Data Pre-processing
Categorical string columns are converted to numerical values so they can be used in the regression model:

- `sex`: `male → 1`, `female → 0`
- `smoker`: `yes → 1`, `no → 0`
- `region`: `northeast → 1`, `southeast → 2`, `southwest → 3`, `northwest → 4` *(1-based, as it is an arbitrary ordinal encoding)*

### 2. Feature Matrix
The six features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) are extracted into a NumPy matrix **x_train** of shape *(1338, 6)*, and the target `charges` column becomes the vector **y_train** of shape *(1338,)*.

### 3. Z-Score Normalisation
Because feature values differ greatly in magnitude, each feature is rescaled using z-score normalisation:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

This ensures gradient descent converges faster and more reliably.

### 4. Linear Regression Model
The model predicts insurance charges using the equation:

$$f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$$

where **w** is the weights vector (one weight per feature) and *b* is the bias/intercept.

### 5. Cost Function
Model performance is measured with the Mean Squared Error (MSE) cost function:

$$J(\mathbf{w},b) = \frac{1}{2m} \sum_{i=0}^{m-1} \left(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\right)^2$$

### 6. Gradient Descent
The weights **w** and bias *b* are optimised iteratively using gradient descent:

$$w_j = w_j - \alpha \frac{\partial J}{\partial w_j}, \quad b = b - \alpha \frac{\partial J}{\partial b}$$

The algorithm runs until the parameters converge or until a maximum of **10,000 iterations** is reached, with a learning rate of **α = 0.01**. Convergence is detected when the weights vector **w** stops changing (checked with `np.allclose`) **and** the scalar bias *b* stops changing (checked with `math.isclose`).

### 7. Results & Visualisation
Two plots are produced to evaluate the model:

- **Cost vs. Iterations** — confirms that gradient descent is converging correctly (cost decreases monotonically).
- **Actual vs. Predicted Charges** — scatter plot of true charges against model predictions, with a red dashed diagonal representing perfect predictions.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix operations, normalisation, gradient computation |
| `pandas` | Loading and pre-processing the CSV dataset |
| `matplotlib` | Visualising cost history and prediction accuracy |
| `math` | Scalar convergence check for bias *b* (`math.isclose`); weights vector convergence uses `np.allclose` |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dulanagit/numpy-linear-regression-medical-costs.git
   cd numpy-linear-regression-medical-costs
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib jupyter
   ```

3. **Launch Jupyter and open the notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

4. **Run all cells** — the model will train, print convergence information, and display the two evaluation plots.

---

## Project Structure

```
numpy-linear-regression-medical-costs/
├── main.ipynb       # Jupyter notebook with the full implementation
├── insurance.csv    # Medical cost dataset
└── README.md        # Project description (this file)
```

---

> **Note:** This project is intended for learning purposes only and implements linear regression from scratch to reinforce understanding of the underlying mathematics.
