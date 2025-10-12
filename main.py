Practical - 1: Implement the KNN algorithm 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['Sepal-length', 'Sepal-width', 'Petal-length', 'Petal-width', 'Species']
data = pd.read_csv(url, names=column_names)
data.head()

data.info()

data.describe()

x = dataset.iloc[:,:-1].values
print(x)

y = dataset.iloc[:,4].values
print(y)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1)

scaler = StandardScaler()
scaler.fit(xtrain)

xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

print(xtrain)
print(xtest)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)
print('Prediction:', y_pred)

print('Confusion Matrix')
cm = confusion_matrix(ytest, y_pred)
print(cm)

print('Classification Report')
print(classification_report(ytest, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for KNN Classifier')
plt.show()

error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain, ytrain)
    pred_i = knn.predict(xtest)
    error.append(np.mean(pred_i != ytest))

plt.figure(figsize=(12,6))
plt.plot(range(1,40), error, color='blue', linestyle='dotted', marker='o', markerfacecolor='green', 
         markersize=10)
plt.title('Error rate VS K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 2: Building Decision Tree Model using the ID3 algorithm using Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split/train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train)
print(Y_train)

scaler = StandardScaler()
xtrain_scaler = scaler.fit_transform(X_train)
xtest_scaler = scaler.transform(X_test)

clf = DecisionTreeClassifier(criterion="entropy")  # entropy implies info gain (ID3 style)
clf.fit(xtrain_scaler, y_train)

ypred = clf.predict(xtest_scaler)
print(ypred)

print("Confusion Matrix")
print(confusion_matrix(ytest, ypred))
print('------------------------------')
print('Classification Report')
print(classification_report(ytest, ypred))

# Export the tree structure for visualization
plt.figure(30,20)
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree For Iris Dataset')
plt.show()
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 3: Developing a Support Vector Machine (SVM) model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
# Assuming dataset CSV file is named 'Social_Network_Ads.csv' and located locally
dataset = pd.read_csv('Social_Network_Ads.csv')

# Select features and target
X = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Linear': SVC(kernel='linear', C=1),
    'Polynomial': SVC(kernel='poly', degree=3, C=1),
    'RBF': SVC(kernel='rbf', gamma='scale', C=1, random_state=42),
    'Sigmoid': SVC(kernel='sigmoid', C=1)
}

for name, model in models.items():
    model.fit(x_train, ytrain)
    ypred = model.predict(x_test)
    print(f'{name} Kernel Classification Report')
    print(classification_report(ytest, ypred))


# Train SVM classifier
svm_classifier = models['RBF"]
svm_classifier.fit(X_train, y_train)


# Plot decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM Decision Boundary - Social Network Ads")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.show()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 4: Building Naive Bayes Classifier

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report

# Step 1: Load and filter data
df = pd.read_csv('/Users/taahashaikh/Downloads/Fish.csv')
df = df[df['Species'].isin(['Bream', 'Perch'])]

# Step 2: Map species directly to numbers (no LabelEncoder)
df['Species'] = df['Species'].map({'Bream': 0, 'Perch': 1})

# Step 3: Split features & target
X = df.drop('Species', axis=1)
y = df['Species']

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 6: Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Step 8: Compute ROC and AUC (only for positive class = 1)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Step 9: Print results
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"\n🔥 AUC Score: {roc_auc:.3f}")

# Step 10: Plot ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve — Perch vs Bream')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
------------------------------------------------------------------------------------------------------------------------------------------------------------------
practical - 5: Implementing Linear Regression
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv('salary.csv')

# Step 2: Split features and target
X = df[['YearsExperience']]  # Feature (2D array)
y = df['Salary']             # Target

# Step 3: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Mean Squared Error:", mse)
print("✅ R^2 Score:", r2)
print("✅ Model Coefficient (slope):", model.coef_[0])
print("✅ Model Intercept:", model.intercept_)

# Step 8: Visualize results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression - Salary vs Experience')
plt.legend()
plt.show()
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 6: Using Logistic Regression on Diabetes Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix,ConfusionMatrixDisplay,classification_report, accuracy_score, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/taahashaikh/Downloads/Diabetes.csv')
# Display first few rows
print(data.head())
print(data.info())

X = data.drop('Outcome', axis=1) # Features
y = data['Outcome']
X
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score (approx): {r2:.2f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Diabetes Prediction')
plt.show()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
practical - 7: Evaluating a classification model using metrices such as accuracy, Precision, recall, and F1-Score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
X
y

y_binary = np.where(y == 0, 1, 0) # 1 = setosa, 0 = not setosa
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate Model Using Metrics
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
# Precision
precision = precision_score(y_test, y_pred)
# Recall
recall = recall_score(y_test, y_pred)
# F1-Score
f1 = f1_score(y_test, y_pred)
# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
# Or use classification_report for a summary
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Setosa','Setosa']))
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 8: Applying hierarchical clustering on iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

sns.set_style('whitegrid')
# 1. Load example data - Iris dataset
iris = datasets.load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward', metric='euclidean')

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='level', p=5, show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Sample index or cluster size')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# 8. Visualize clusters in 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
palette = sns.color_palette('deep', n_colors=k)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_sklearn,palette=palette, legend='full', s=60)
plt.title(f'Agglomerative Clustering k={k} PCA projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 9: K means clustering
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)

test_df['Survived'] = -1  # Dummy value

df = pd.concat([train_df, test_df], ignore_index=True)

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].copy()

X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

df['Cluster'] = kmeans.labels_

print(df['Cluster'].value_counts())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters Visualized with PCA')
plt.colorbar(label='Cluster')
plt.show()
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical - 10:  Utilizing principal component analysis (PCA) for dimensionality reduction to improve the eﬀiciency and interpretability of a model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = data.data
y = data.target 
target_names = data.target_names
X
y
target_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio of each component:", explained_variance)
print("Total variance explained by 10 components:", explained_variance.sum())

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Target'] = y

plt.figure(figsize=(8,6))
colors = ['red', 'blue']
for target in [0,1]:
    subset = df_pca[df_pca['Target'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], 
                label=target_names[target], alpha=0.7, s=50, c=colors[target])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
----------------------------------------------------------------- THE END ------------------------------------------------------------------------------------------
