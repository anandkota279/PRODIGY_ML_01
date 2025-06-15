import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('/content/train.csv')

print("Dataset shape:",dataset.shape)
print("\nDataset description:\n",dataset.describe())
print("\nDataset head:\n",dataset.head())
print("\nDataset tail:\n",dataset.tail())
print("\nMissing values:\n", dataset.isnull().sum())

dataset['TotalBath'] = dataset['FullBath'] + 0.5 * dataset['HalfBath'] 
features=dataset[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
target = dataset['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))
print("MSE", metrics.mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(dataset['SalePrice'], kde=True, color='skyblue', bins=40)
plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


sorted_idx = np.argsort(y_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='gray', label='Predictions')
plt.plot(y_test.iloc[sorted_idx], y_pred[sorted_idx], color='red', linewidth=2, label='Trend Line')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted with Trend Line")
plt.legend()
plt.show()


comparison_df = pd.DataFrame({'Actual': y_test.values[:25], 'Predicted': y_pred[:25]})
comparison_df.plot(kind='bar', figsize=(15, 6))
plt.title("Actual vs Predicted Prices (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Price")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()


