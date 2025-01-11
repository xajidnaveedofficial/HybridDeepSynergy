# HybridDeepSynergy
A Hybrid Deep Learning Model Integrating CNN, LSTM, and Attention Mechanisms for Cancer Drug Synergy Prediction

# Usage:
# Step 1: Clone the Repository
Clone this repository to your local machine:
 git clone https://github.com/your-username/drug-synergy-predictio](https://github.com/xajidnaveedofficial/HybridDeepSynergy.git
cd drug-synergy-prediction
```python 
your_code = do_some_stuff
```
# Step 2: Load the Dataset
Replace data_URL with the path to your dataset in the provided code:

```python 
import pandas as pd
df_data = pd.read_csv(data_URL)

```
# Step 3: Preprocess Data
The code includes functions to preprocess gene expression data, drug information, and cell line data for modeling.

# Step 4: Train the Model
Train the model using the prepared dataset:

```python 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)
```
# Step 5: Evaluate the Model
Evaluate the model's performance using metrics such as MSE, RMSE, MAE, Pearson, and Spearman correlations:
```python 
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Square (R2): {r2:.4f}")

```






