import bentoml
from sklearn import datasets, svm

# Load the dataset
X, y = datasets.load_iris(return_X_y=True)

# Model Training
clf = svm.SVC(gamma="scale")
clf.fit(X, y)

# Create a BentoService
saved_model = bentoml.sklearn.save_model("iris_svm_clf", clf)
print(f"Model saved: {saved_model}")
