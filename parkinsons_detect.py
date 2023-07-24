import pandas as pd
import pickle
import numpy as np

# Load the data from CSV files
df1 = pd.read_csv("stage1.csv", nrows=10000)
df2 = pd.read_csv("stage2.csv", nrows=10000)
df3 = pd.read_csv("stage3.csv", nrows=10000)
df4 = pd.read_csv("stage4.csv", nrows=10000)

# Concatenate the data into a single DataFrame
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

print(df)

#preprocessing 
# Check for missing values
if df.isnull().values.any():
    df = df.dropna()

# Check for infinite values
if np.isinf(df.values).any():
    df = np.nan_to_num(df, nan=0, posinf=1e+20, neginf=-1e+20)

# Check the data type
if df.values.dtype != np.float64:
    df = df.astype(np.float64)

# Separate the input (features) and output (target) variables
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.preprocessing import StandardScaler

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN model
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)

ypred = knn.predict(X_test)


# Evaluate the performance of the KNN model on the testing set
score = knn.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(score * 100))


# Save your KNN model as a pickle file
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)