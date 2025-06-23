import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate a binary classification dataset with 1 feature
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Visualize raw data
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k')
plt.title("Generated Binary Data")
plt.xlabel("Feature")
plt.ylabel("Class")
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate input range for sigmoid
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]  # Probability of class 1

# Plot sigmoid
plt.figure(figsize=(8, 5))
plt.plot(X_range, y_prob, color='red', linewidth=2, label='Sigmoid Curve (P=1)')
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k', alpha=0.6, label='Actual Data')
plt.title("Logistic Regression - Sigmoid Curve")
plt.xlabel("Feature")
plt.ylabel("Probability / Class")
plt.legend()
plt.grid(True)
plt.show()
