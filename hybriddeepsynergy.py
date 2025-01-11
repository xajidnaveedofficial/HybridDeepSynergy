data_URL = ('/experment/dataset/drugcomb.csv')

import pandas as pd
df_data = pd.read_csv(data_URL)

df_data.head()

!pip install rdkit-pypi shap tensorflow

# #######################################Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM, MultiHeadAttention, LayerNormalization, Add, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Import the necessary functions from scipy.stats
from scipy.stats import pearsonr, spearmanr

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from keras.models import Model
# from keras.layers import Input, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout, Add, LayerNormalization, MultiHeadAttention
# from keras.optimizers import Adam
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming df_data is already defined and contains your data
df = pd.DataFrame(df_data)
#col = df.drop(['CSS', 'S',], axis=1)
# Define your target variable(s)
#target_columns = ['CSS', 'S', 'ZIP', 'BLISS', 'LOEWE', 'HSA']
target_columns = ['S','ZIP', 'BLISS', 'LOEWE', 'HSA']

y = df[target_columns]

# Drop the target columns from the features DataFrame
X = df.drop(columns=target_columns)

# One-hot encode categorical variables
feature_columns = X.select_dtypes(include=['number']).columns.tolist()  # Get existing numerical features
for col in ['DrugA', 'DrugB', 'Cell_line']:
    if col in X.columns:
        one_hot = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = pd.concat([X, one_hot], axis=1)
        X.drop(columns=[col], inplace=True)
        feature_columns.extend(one_hot.columns)

# Now, proceed with the train-test split
X_train, X_test, y_train, y_test = train_test_split(X[feature_columns], y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the input data to be 3D for LSTM and Attention layers
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Adjusted model architecture and training
#input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
# Adjusted model architecture to ensure compatibility
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
cnn = Conv1D(filters=128, kernel_size=1, activation='relu')(input_layer) # Changed filter size to 128 to match LSTM output
cnn = GlobalMaxPooling1D()(cnn)

lstm = LSTM(256, return_sequences=True)(input_layer)
lstm = LSTM(128)(lstm)

attention_output = MultiHeadAttention(num_heads=8, key_dim=32, attention_axes=(1,))(lstm, lstm)
attention_output = Add()([attention_output, lstm])
attention_output = LayerNormalization()(attention_output)

combined = Add()([cnn, attention_output]) # Now, cnn and attention_output have the same shape (None, 128)
dense1 = Dense(512, activation='relu', kernel_regularizer='l2')(combined)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(256, activation='relu', kernel_regularizer='l2')(dropout1)
dropout2 = Dropout(0.3)(dense2)

output = Dense(y_train.shape[1], activation='linear')(dropout2)
model = Model(inputs=input_layer, outputs=output)
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.003), loss='mean_squared_error', metrics=['mae'])

# Callbacks for early stopping and learning rate adjustment
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
learning_rate_schedule = LearningRateScheduler(lambda epoch: 0.001 * 0.9 ** epoch)



model.fit(
    X_train, y_train, validation_data=(X_test, y_test),
    epochs=50, batch_size=8,
    callbacks=[early_stopping, learning_rate_schedule]
)


# Predict on test data
y_pred = model.predict(X_test)

# Convert y_pred to DataFrame for easier analysis and comparison
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)
print(y_pred_df.head())

# Calculate MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Root Square (R2): {r2:.4f}")

# from scipy.stats import pearsonr, spearmanr
# import numpy as np

# # Flatten all actual and predicted values
# y_test_flattened = y_test_df.values.flatten()
# y_pred_flattened = y_pred_df.values.flatten()

# # Calculate overall Pearson and Spearman correlation
# overall_pearson_corr, _ = pearsonr(y_test_flattened, y_pred_flattened)
# overall_spearman_corr, _ = spearmanr(y_test_flattened, y_pred_flattened)

# print("Overall Pearson Correlation:", overall_pearson_corr)
# print("Overall Spearman Correlation:", overall_spearman_corr)


# Initialize lists to store correlations for each column
pearson_corr = []  # Initialize as a list
spearman_corr = [] # Initialize as a list

for i in range(y_test.shape[1]):  # Iterate through columns (target variables)
    # Extract the i-th column from y_test and y_pred
    y_test_col = y_test.iloc[:, i].values
    y_pred_col = y_pred[:, i]

    # Calculate correlation for the current column
    corr, _ = pearsonr(y_test_col, y_pred_col)
    pearson_corr.append(corr)

    corr, _ = spearmanr(y_test_col, y_pred_col)
    spearman_corr.append(corr)

# Print or use the correlation values as needed
print(f"Pearson correlations: {pearson_corr}")
print(f"Spearman correlations: {spearman_corr}")

# Convert true and predicted values to DataFrames if necessary
y_test_df = pd.DataFrame(y_test, columns=y.columns)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

# Calculate MSE, RMSE, MAE, R², Pearson, and Spearman for each output column
metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'Pearson': [], 'Spearman': []}
for col in y.columns:
    mse = mean_squared_error(y_test_df[col], y_pred_df[col])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_df[col], y_pred_df[col])
    r2 = r2_score(y_test_df[col], y_pred_df[col])
    pearson_corr, _ = pearsonr(y_test_df[col], y_pred_df[col])
    spearman_corr, _ = spearmanr(y_test_df[col], y_pred_df[col])

    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['MAE'].append(mae)
    metrics['R2'].append(r2)
    metrics['Pearson'].append(pearson_corr)
    metrics['Spearman'].append(spearman_corr)

    print(f"{col} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

import pandas as pd

# Convert y_test and y_pred to DataFrames for easier analysis
y_test_df = pd.DataFrame(y_test, columns=y.columns)  # Ground truth values
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)  # Predicted values

# Combine actual and predicted values for comparison
results_df = pd.concat([y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
results_df.columns = [f"Actual_{col}" for col in y.columns] + [f"Predicted_{col}" for col in y.columns]

# Display the results
print(results_df.head())  # Show the first few rows for quick inspection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Convert true and predicted values to DataFrames if necessary
y_test_df = pd.DataFrame(y_test, columns=y.columns)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

# Calculate MSE, RMSE, MAE, R², Pearson, and Spearman for each output column
metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': [], 'Pearson': [], 'Spearman': []}
for col in y.columns:
    mse = mean_squared_error(y_test_df[col], y_pred_df[col])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_df[col], y_pred_df[col])
    r2 = r2_score(y_test_df[col], y_pred_df[col])
    pearson_corr, _ = pearsonr(y_test_df[col], y_pred_df[col])
    spearman_corr, _ = spearmanr(y_test_df[col], y_pred_df[col])

    metrics['MSE'].append(mse)
    metrics['RMSE'].append(rmse)
    metrics['MAE'].append(mae)
    metrics['R2'].append(r2)
    metrics['Pearson'].append(pearson_corr)
    metrics['Spearman'].append(spearman_corr)

    print(f"{col} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

# Display metrics as a DataFrame
metrics_df = pd.DataFrame(metrics, index=y.columns)
print(metrics_df)

# Plotting MSE and MAE
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
metrics_df[['MSE', 'MAE']].plot(kind='bar', ax=ax[0])
ax[0].set_title("MSE and MAE for each target variable")
ax[0].set_xlabel("Target Variable")
ax[0].set_ylabel("Error")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(metrics_df[['Pearson', 'Spearman']].T, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation between Predictions and True Values")
plt.show()

# Scatter plots for each target to visualize true vs predicted values
for col in y.columns:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_df[col], y_pred_df[col], alpha=0.5)
    plt.plot([y_test_df[col].min(), y_test_df[col].max()], [y_test_df[col].min(), y_test_df[col].max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted Values for {col}")
    plt.show()

# Assuming you have a DataFrame `df` containing synergy metrics
plt.figure(figsize=(10, 8))
sns.heatmap(df[[ 'S','ZIP', 'BLISS', 'LOEWE', 'HSA']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Drug Synergy Metrics')
plt.show()

sns.pairplot(df[['S','ZIP', 'BLISS', 'LOEWE', 'HSA']], diag_kind='kde', plot_kws={'alpha':0.7})
plt.title('Pair Plot of Drug Combination Synergy Metrics')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming `drug_info_test` contains columns 'DrugA' and 'DrugB' for each sample in the test set
drug_info_test = df[['DrugA', 'DrugB']].iloc[y_test.index]  # Retrieve based on y_test indices

# Convert y_test and y_pred to DataFrames for easier analysis
y_test_df = pd.DataFrame(y_test, columns=target_columns)  # Actual values
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)  # Predicted values

# Combine drug information with actual and predicted values
df_test = pd.concat([drug_info_test.reset_index(drop=True), y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
df_test.columns = ['DrugA', 'DrugB'] + [f'Actual_{col}' for col in target_columns] + [f'Predicted_{col}' for col in target_columns]

# Define the number of drug combinations to display on each plot
combinations_to_display = 10  # Adjust this number as needed for readability

# Generate unique color for each drug combination
unique_combinations = df_test['DrugA'] + " + " + df_test['DrugB']
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_combinations.unique())))
color_map = dict(zip(unique_combinations.unique(), colors))

# Loop over each metric and create one plot with a reduced number of combinations
for metric in target_columns:
    fig, ax = plt.subplots(figsize=(14, 8))  # Wide layout for readability

    # Select a limited subset of drug combinations for the plot
    df_subset = df_test.sample(n=combinations_to_display, random_state=1)  # Random sample for variety; adjust if needed

    # Construct labels and plot bars with color coding
    labels = df_subset['DrugA'] + " + " + df_subset['DrugB']
    for i, combination in enumerate(labels):
        actual_value = df_subset[f'Actual_{metric}'].iloc[i]
        predicted_value = df_subset[f'Predicted_{metric}'].iloc[i]

        # Plot bars with unique colors for each drug combination
        ax.barh(
            combination,
            actual_value,
            color=color_map[combination],
            label=combination if combination not in ax.get_legend_handles_labels()[1] else "",
            height=0.5
        )
        ax.barh(
            combination,
            predicted_value,
            color=color_map[combination],
            alpha=0.5,
            height=0.4
        )

    # Set title, labels, and font sizes
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Actual vs. Predicted {metric} for Selected Drug Combinations', fontsize=14)

    # Legend setup to display color-coded drug combinations
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Drug Combinations", bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fontsize=8, title_fontsize=9, frameon=False)

    # Adjust y-axis labels for readability and apply grid
    ax.tick_params(axis='y', labelsize=10)  # Adjusted label size for readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Tight layout and save plot per page
    plt.tight_layout(pad=3)
    plt.savefig(f"actual_vs_predicted_{metric}_drugs_reduced.png", dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming `drug_info_test` contains columns 'DrugA', 'DrugB', and 'CellLine' for each sample in the test set
# Adjust data to include cell line information
drug_info_test = df[['DrugA', 'DrugB', 'Cell_line']].iloc[y_test.index]  # Retrieve based on y_test indices

# Convert y_test and y_pred to DataFrames for easier analysis
y_test_df = pd.DataFrame(y_test, columns=target_columns)  # Actual values
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)  # Predicted values

# Combine drug information with actual and predicted values
df_test = pd.concat([drug_info_test.reset_index(drop=True), y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
df_test.columns = ['DrugA', 'DrugB', 'Cell_line'] + [f'Actual_{col}' for col in target_columns] + [f'Predicted_{col}' for col in target_columns]

# Define the number of combinations to display on each plot
combinations_to_display = 10  # Adjust this number as needed for readability

# Generate unique colors for each drug-cell line combination
unique_combinations = df_test['DrugA'] + " + " + df_test['DrugB'] + " (" + df_test['Cell_line'] + ")"
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_combinations.unique())))
color_map = dict(zip(unique_combinations.unique(), colors))

# Loop over each metric and create one plot with a reduced number of combinations
for metric in target_columns:
    fig, ax = plt.subplots(figsize=(14, 8))  # Wide layout for readability

    # Select a limited subset of drug-cell line combinations for the plot
    df_subset = df_test.sample(n=combinations_to_display, random_state=1)  # Random sample for variety

    # Construct labels that include drugs and cell line, and plot bars with color coding
    labels = df_subset['DrugA'] + " + " + df_subset['DrugB'] + " (" + df_subset['Cell_line'] + ")"
    for i, combination in enumerate(labels):
        actual_value = df_subset[f'Actual_{metric}'].iloc[i]
        predicted_value = df_subset[f'Predicted_{metric}'].iloc[i]

        # Plot bars with unique colors for each drug-cell line combination
        ax.barh(
            combination,
            actual_value,
            color=color_map[combination],
            label=combination if combination not in ax.get_legend_handles_labels()[1] else "",
            height=0.5
        )
        ax.barh(
            combination,
            predicted_value,
            color=color_map[combination],
            alpha=0.5,
            height=0.4
        )

    # Set title, labels, and font sizes
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'Actual vs. Predicted {metric} for Selected Drug Combinations and Cell Lines', fontsize=14)

    # Legend setup to display color-coded drug-cell line combinations
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Drug Combinations (with Cell Line)", bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fontsize=8, title_fontsize=9, frameon=False)

    # Adjust y-axis labels for readability and apply grid
    ax.tick_params(axis='y', labelsize=10)  # Adjusted label size for readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Tight layout and save plot per page
    plt.tight_layout(pad=3)
    plt.savefig(f"actual_vs_predicted_{metric}_drugs_cell_line.png", dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming `drug_info_test` contains columns 'DrugA' and 'DrugB' for each sample in the test set
# If not, retrieve it from your initial data if it's not already present.
drug_info_test = df[['DrugA', 'DrugB']].iloc[y_test.index]  # Retrieve based on y_test indices

# Convert y_test and y_pred to DataFrames for easier analysis
y_test_df = pd.DataFrame(y_test, columns=target_columns)  # Actual values
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)  # Predicted values

# Combine drug information with actual and predicted values
df_test = pd.concat([drug_info_test.reset_index(drop=True), y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
df_test.columns = ['DrugA', 'DrugB'] + [f'Actual_{col}' for col in target_columns] + [f'Predicted_{col}' for col in target_columns]

# Plot Actual vs Predicted for each synergy metric
for metric in target_columns:
    plt.figure(figsize=(10, 24))
    plt.barh(df_test['DrugA'] + " + " + df_test['DrugB'], df_test[f'Actual_{metric}'], color='blue', label=f'Actual {metric}')
    plt.barh(df_test['DrugA'] + " + " + df_test['DrugB'], df_test[f'Predicted_{metric}'], color='orange', alpha=0.7, label=f'Predicted {metric}')
    plt.xlabel(metric)
    plt.title(f'Actual vs. Predicted {metric} for Drug Combinations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"actual_vs_predicted_{metric}_drugs.png", dpi=300, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming `drug_info_test` contains columns 'DrugA' and 'DrugB' for each sample in the test set
# If not, retrieve it from your initial data if it's not already present.
drug_info_test = df[['DrugA', 'DrugB']].iloc[y_test.index]  # Retrieve based on y_test indices

# Convert y_test and y_pred to DataFrames for easier analysis
y_test_df = pd.DataFrame(y_test, columns=target_columns)  # Actual values
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)  # Predicted values

# Combine drug information with actual and predicted values
df_test = pd.concat([drug_info_test.reset_index(drop=True), y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
df_test.columns = ['DrugA', 'DrugB'] + [f'Actual_{col}' for col in target_columns] + [f'Predicted_{col}' for col in target_columns]

# Plot Actual vs Predicted for each synergy metric
for metric in target_columns:
    fig, ax = plt.subplots(figsize=(10, 24))  # Wider layout for readability

    # Construct labels and plot bars with adjusted spacing and font sizes
    labels = df_test['DrugA'] + " + " + df_test['DrugB']
    ax.barh(labels, df_test[f'Actual_{metric}'], color='blue', label=f'Actual {metric}', height=0.5)
    ax.barh(labels, df_test[f'Predicted_{metric}'], color='orange', alpha=0.7, label=f'Predicted {metric}', height=0.4)

    # Set title and labels
    ax.set_xlabel(metric, fontsize=10)  # Smaller font for value labels
    ax.set_title(f'Actual vs. Predicted {metric} for Drug Combinations', fontsize=14)
    ax.legend()

    # Adjust drug label font size for better readability
    ax.tick_params(axis='y', labelsize=12)  # Larger font size for drug labels
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Increase spacing and save the plot
    plt.tight_layout(pad=3)
    plt.savefig(f"actual_vs_predicted_{metric}_drugs.png", dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# Convert true and predicted values to DataFrames if necessary
y_test_df = pd.DataFrame(y_test, columns=y.columns)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

# Calculate MSE, MAE, and Correlation for each output column
metrics = {'MSE': [], 'MAE': [], 'Correlation': []}
for col in y.columns:
    mse = mean_squared_error(y_test_df[col], y_pred_df[col])
    mae = mean_absolute_error(y_test_df[col], y_pred_df[col])
    correlation = np.corrcoef(y_test_df[col], y_pred_df[col])[0, 1]

    metrics['MSE'].append(mse)
    metrics['MAE'].append(mae)
    metrics['Correlation'].append(correlation)

    print(f"{col} - MSE: {mse:.4f}, MAE: {mae:.4f}, Correlation: {correlation:.4f}")

# Display metrics as a DataFrame
metrics_df = pd.DataFrame(metrics, index=y.columns)
print(metrics_df)

# Plotting MSE and MAE
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
metrics_df[['MSE', 'MAE']].plot(kind='bar', ax=ax[0])
ax[0].set_title("MSE and MAE for each target variable")
ax[0].set_xlabel("Target Variable")
ax[0].set_ylabel("Error")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(metrics_df[['Correlation']].T, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation between Predictions and True Values")
plt.show()

# Scatter plots for each target to visualize true vs predicted values
for col in y.columns:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_df[col], y_pred_df[col], alpha=0.5)
    plt.plot([y_test_df[col].min(), y_test_df[col].max()], [y_test_df[col].min(), y_test_df[col].max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted Values for {col}")
    plt.show()

import matplotlib.pyplot as plt

# Loop over each target variable to create the true vs predicted graph
for col in y.columns:
    plt.figure(figsize=(10, 6))

    # Plot true values
    plt.plot(y_test_df[col].values, label='True Values', color='blue')

    # Plot predicted values
    plt.plot(y_pred_df[col].values, label='Predicted Values', color='orange', alpha=0.7)

    plt.title(f"True vs Predicted Values for {col}")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming y_test_df and y_pred_df have already been created
# Combine the data into one DataFrame for easier plotting
results_df = pd.concat([y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
results_df.columns = [f"Actual_{col}" for col in y_test_df.columns] + [f"Predicted_{col}" for col in y_pred_df.columns]

# Scatter plots for actual vs. predicted values for each metric
plt.figure(figsize=(16, 10))
for i, col in enumerate(y_test_df.columns, 1):
    plt.subplot(2, 3, i)  # Adjust the grid layout based on the number of columns
    plt.scatter(results_df[f"Actual_{col}"], results_df[f"Predicted_{col}"], alpha=0.5, color='b')
    plt.plot([results_df[f"Actual_{col}"].min(), results_df[f"Actual_{col}"].max()],
             [results_df[f"Actual_{col}"].min(), results_df[f"Actual_{col}"].max()],
             color='r', linestyle='--')  # Line of perfect prediction
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"Actual vs. Predicted {col}")
plt.tight_layout()
plt.show()

# Correlation heatmaps for Pearson and Spearman correlations
# Calculate Pearson and Spearman correlations
pearson_corr = results_df.corr(method='pearson')
spearman_corr = results_df.corr(method='spearman')

# Plot Pearson correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Pearson Correlation Heatmap")
plt.show()

# Plot Spearman correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Spearman Correlation Heatmap")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Residual plot for each metric
plt.figure(figsize=(16, 10))
for i, col in enumerate(y_test_df.columns, 1):
    residuals = results_df[f"Actual_{col}"] - results_df[f"Predicted_{col}"]
    plt.subplot(2, 3, i)
    plt.scatter(results_df[f"Actual_{col}"], residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Residuals {col}")
    plt.title(f"Residual Plot for {col}")
plt.tight_layout()
plt.show()

# Error Distribution (Histogram of residuals)
plt.figure(figsize=(16, 10))
for i, col in enumerate(y_test_df.columns, 1):
    residuals = results_df[f"Actual_{col}"] - results_df[f"Predicted_{col}"]
    plt.subplot(2, 3, i)
    sns.histplot(residuals, kde=True, color='skyblue')
    plt.title(f"Error Distribution for {col}")
    plt.xlabel("Residual Error")
plt.tight_layout()
plt.show()

# Train the model and save the training history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16)

# Learning Curve (requires access to model's training history)
# Assuming `history` is the model's training history from model.fit()
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

# R² Scores for each metric
r2_scores = {col: r2_score(results_df[f"Actual_{col}"], results_df[f"Predicted_{col}"]) for col in y_test_df.columns}
print("R² Scores per Metric:")
for metric, score in r2_scores.items():
    print(f"{metric}: {score:.4f}")

# Visualization of R² scores
plt.figure(figsize=(8, 5))
plt.bar(r2_scores.keys(), r2_scores.values(), color='skyblue')
plt.xlabel('Metrics')
plt.ylabel('R² Score')
plt.title('R² Score for Each Synergy Metric')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter Plot for each synergy metric
for col in y.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=results_df[f"Actual_{col}"], y=results_df[f"Predicted_{col}"])
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"Scatter Plot of Actual vs. Predicted for {col}")
    plt.plot([results_df[f"Actual_{col}"].min(), results_df[f"Actual_{col}"].max()],
             [results_df[f"Actual_{col}"].min(), results_df[f"Actual_{col}"].max()],
             'r--')  # Diagonal reference line
    plt.show()

# Select a subset of data for demonstration (top 10 rows)
subset_df = results_df.head(10)
cell_lines = df['Cell_line'].head(10)  # Assuming 'Cell_line' column was included before preprocessing

# Plotting Actual vs Predicted for each synergy metric per cell line
for col in y.columns:
    plt.figure(figsize=(12, 8))
    subset_actual = subset_df[[f"Actual_{col}"]].reset_index(drop=True)
    subset_predicted = subset_df[[f"Predicted_{col}"]].reset_index(drop=True)
    combined = pd.concat([subset_actual, subset_predicted], axis=1)
    combined['Cell_line'] = cell_lines

    combined.plot(kind='bar', width=0.8, figsize=(12, 6))
    plt.xticks(ticks=range(len(combined['Cell_line'])), labels=combined['Cell_line'], rotation=45)
    plt.title(f"Comparison of Actual and Predicted {col} Scores Across Cell Lines")
    plt.xlabel("Cell Line")
    plt.ylabel(f"{col} Score")
    plt.legend(['Actual', 'Predicted'])
    plt.show()

# Calculate correlation matrix for Actual vs Predicted columns
correlation_matrix = results_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Actual and Predicted Scores")
plt.show()

#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEENNNNNNNNNNNNNNNNNNNNNNNDDDDDDDDDDDDDDd

#EEEEEEEEEEEEEEEEEEEEEEEEEEEEEENNNNNNNNNNNNNNNNNNNNNNNDDDDDDDDDDDDDDd
#

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming `drug_info_test` contains columns 'DrugA' and 'DrugB' for each sample in the test set
drug_info_test = df[['DrugA', 'DrugB']].iloc[y_test.index]  # Retrieve based on y_test indices

# Convert y_test and y_pred to DataFrames for easier analysis
y_test_df = pd.DataFrame(y_test, columns=target_columns)  # Actual values
y_pred_df = pd.DataFrame(y_pred, columns=target_columns)  # Predicted values

# Combine drug information with actual and predicted values
df_test = pd.concat([drug_info_test.reset_index(drop=True), y_test_df.reset_index(drop=True), y_pred_df.reset_index(drop=True)], axis=1)
df_test.columns = ['DrugA', 'DrugB'] + [f'Actual_{col}' for col in target_columns] + [f'Predicted_{col}' for col in target_columns]

# Define how many combinations to plot per page for readability
combinations_per_page = 10  # Adjust this to fit your layout preferences

# Generate unique color for each drug combination
unique_combinations = df_test['DrugA'] + " + " + df_test['DrugB']
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_combinations.unique())))
color_map = dict(zip(unique_combinations.unique(), colors))

# Loop over each metric and create separate figures for batches of drug combinations
for metric in target_columns:
    num_pages = int(np.ceil(len(df_test) / combinations_per_page))

    for page in range(num_pages):
        fig, ax = plt.subplots(figsize=(14, 8))  # Set a wide figure size for readability

        # Select subset of drug combinations for the current page
        start_idx = page * combinations_per_page
        end_idx = start_idx + combinations_per_page
        df_subset = df_test.iloc[start_idx:end_idx]

        # Construct labels and plot bars with adjusted spacing and font sizes
        labels = df_subset['DrugA'] + " + " + df_subset['DrugB']
        for i, combination in enumerate(labels):
            actual_value = df_subset[f'Actual_{metric}'].iloc[i]
            predicted_value = df_subset[f'Predicted_{metric}'].iloc[i]

            # Plot bars with different colors for each drug combination
            ax.barh(
                combination,
                actual_value,
                color=color_map[combination],
                label=combination if combination not in ax.get_legend_handles_labels()[1] else "",
                height=0.5
            )
            ax.barh(
                combination,
                predicted_value,
                color=color_map[combination],
                alpha=0.5,
                height=0.4
            )

        # Set title and labels
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Actual vs. Predicted {metric} for Drug Combinations (Page {page + 1}/{num_pages})', fontsize=14)

        # Legend at the bottom with reduced font size
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Drug Combinations", bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3, fontsize=8, title_fontsize=9, frameon=False)

        # Adjust y-axis labels for readability and apply grid
        ax.tick_params(axis='y', labelsize=10)  # Adjusted label size for readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Tight layout and save plot per page
        plt.tight_layout(pad=3)
        plt.savefig(f"actual_vs_predicted_{metric}_drugs_page_{page + 1}.png", dpi=300, bbox_inches='tight')
        plt.show()

# Assuming `combination_names` is a list of drug combinations and `synergy_scores` are the predicted scores
# Example:
combination_names = ['DrugA+DrugB', 'DrugC+DrugD', 'DrugE+DrugF']  # Replace with your actual drug combinations
synergy_scores = [0.8, 0.5, 0.2]  # Replace with your predicted synergy scores

plt.figure(figsize=(12, 6))
sns.barplot(x=combination_names, y=synergy_scores, palette='coolwarm')
plt.xticks(rotation=90)
plt.xlabel('Drug Combinations')
plt.ylabel('Predicted Synergy Score')
plt.title('Predicted Synergy Scores for Drug Combinations')
plt.show()

# Importing the required libraries
import matplotlib.pyplot as plt
import numpy as np

# Original long drug names list
drug_names = ['hydroxyurea' + 'cyclophosphamide', 'Triethylenemelamine' + 'Vinblastine sulfate',
              'Vinblastine sulfate'    +     'CHEMBL277800',
              'letrozole'    +      'MITHRAMYCIN', 'celecoxib' + 'Antibiotic AY 22989']

# Function to wrap drug names into multiple lines
def wrap_labels(labels, width):
    return ['\n'.join(label[i:i+width] for i in range(0, len(label), width)) for label in labels]

# Shorten long names by breaking them into multiple lines
wrapped_drug_names = wrap_labels(drug_names, 20)  # Adjust width accordingly

# Actual and predicted values for the drugs
actual_values = [5.28 ,22.51, 6.16, 15.66, 9.76]  # Replace with actual values from your dataset
predicted_values = [ 5.44 ,23.04, 5.85, 15.62, 10.30]  # Replace with your model's predicted values

# Creating a bar plot for actual vs predicted values
fig, ax = plt.subplots(figsize=(10,6))

# Define bar width and positions
bar_width = 0.35
index = np.arange(len(drug_names))

# Bar plots for actual and predicted values
actual_bar = ax.bar(index, actual_values, bar_width, label='Actual', color='b')
predicted_bar = ax.bar(index + bar_width, predicted_values, bar_width, label='Predicted', color='r')

# Adding labels and title
ax.set_xlabel('Drugs')
ax.set_ylabel('Sensitivity (IC50)')
ax.set_title('Actual vs Predicted Drug Sensitivity')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(wrapped_drug_names, rotation=45, ha='right')  # Rotate labels for better visibility

# Adding legend
ax.legend()

# Function to add value labels on top of each bar
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Add value labels to both bars
add_value_labels(actual_bar)
add_value_labels(predicted_bar)

# Display the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(drug_names, actual_values, label='Actual', color='blue', marker='o')
plt.plot(drug_names, predicted_values, label='Predicted', color='red', marker='x')
plt.xlabel('Drugs')
plt.ylabel('Sensitivity (IC50)')
plt.title('Actual vs Predicted Drug Sensitivity (Line Plot)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

errors = np.array(actual_values) - np.array(predicted_values)

plt.figure(figsize=(10, 6))
plt.boxplot(errors, vert=False)
plt.title('Error Distribution (Actual - Predicted Sensitivity)')
plt.xlabel('Error')
plt.tight_layout()
plt.show()

# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Creating the dataset with the actual and predicted values
data = {
    'Drug Combinations': ['hydroxyurea' + 'cyclophosphamide', 'Triethylenemelamine' + 'Vinblastine sulfate',
              'Vinblastine sulfate'    +     'CHEMBL277800',
              'letrozole'    +      'MITHRAMYCIN', 'celecoxib' + 'Antibiotic AY 22989'],
    'Actual Sensitivity': [ 5.28 ,22.51, 6.16, 15.66, 9.76],
    'Predicted Sensitivity': [5.44 ,23.04, 5.85, 15.62, 10.30]
}

# Creating a DataFrame from the dictionary
df = pd.DataFrame(data)

# Plotting a pair plot
sns.pairplot(df, kind="scatter")

# Adding a title
plt.suptitle("Pair Plot of Actual vs Predicted Drug Sensitivity", y=1.02)

# Show the plot
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assume your data has columns 'CSS/RI', 'S', 'ZIP', 'BLISS', 'LOEWE', 'HSA', and the predicted values
# Ensuring all columns have the same length (4 in this case)
df_correlation = pd.DataFrame({
    'CSS': [3.19, 3.42, 15.88, 33.35],
    'S': [-3.39, 5.22, 5.81, 22.04],
    'ZIP': [-3.23, 0.48, -4.75, 8.48],
    'BLISS': [-0.17, 3.7, -2.17, 11.51],
    'LOEWE': [-4.85, 1.3, -0.85, -14.59],
    'HAS': [-0.77, 1.1, -0.71, 12.36],
    'Predicted Sensitivity': predicted_values[:4] # Taking only the first 4 predicted values to match length
})

plt.figure(figsize=(10, 6))
sns.heatmap(df_correlation.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap Between Metrics and Predicted Sensitivity')
plt.show()

# Import necessary libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Plot Actual vs. Predicted CSS/RI Values
plt.figure(figsize=(8, 6))
plt.scatter(df_test['CSS'], df_test['Predicted_CSS_RI'], color='blue', label='Predicted vs Actual')
plt.plot([df_test['CSS'].min(), df_test['CSS'].max()],
         [df_test['CSS'].min(), df_test['CSS'].max()], 'r--', label='Ideal fit (y=x)')
plt.xlabel('Actual CSS/RI')
plt.ylabel('Predicted CSS/RI')
plt.title('Actual vs. Predicted CSS/RI')
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches='tight')  # Save the plot for publication
plt.show()

# 2. Plot Residuals (Errors)
residuals = df_test['CSS'] - df_test['Predicted_CSS_RI']

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='green')
plt.axvline(0, color='black', linestyle='--')
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Density')
plt.grid(True)
plt.savefig("residuals_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Loss Curves: Training vs. Validation Loss
# history = model.history  # Assuming the history object is captured during model training

# plt.figure(figsize=(8, 6))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig("loss_curves.png", dpi=300, bbox_inches='tight')
# plt.show()

# 4. Plot Actual vs Predicted with Drug Names
plt.figure(figsize=(10, 24))
plt.barh(df_test['DrugA'] + " + " + df_test['DrugB'], df_test['CSS'], color='blue', label='Actual CSS/RI')
plt.barh(df_test['DrugA'] + " + " + df_test['DrugB'], df_test['Predicted_CSS_RI'], color='orange', alpha=0.7, label='Predicted CSS/RI')
plt.xlabel('CSS/RI')
plt.title('Actual vs. Predicted CSS/RI for Drug Combinations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_drugs.png", dpi=300, bbox_inches='tight')
plt.show()