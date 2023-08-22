# Iris Classification Project using Support Vector Machine (SVM) and BentoML

This project focuses on the development of a machine learning model using the Iris dataset. We use Support Vector Machine (SVM) as our model of choice which we implemented using sklearn. The model is built, trained, and then served using BentoML which provides an incredible way of handling models in production.

## Project Set Up

The project setup is simple and straightforward. You need to have [Python](https://www.python.org/downloads/), BentoML and Scikit-Learn installed. To install BentoML and Scikit-learn, run the following command:

```bash
pip install bentoml scikit-learn
```

## Building the Model

The model is built using the Iris dataset. SVM is chosen for this multi-class classification problem. The dataset contains 150 instances of iris flowers from three different species.

Here's an overview of how the model was saved:

```python
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
```

The `save_model` function is used to save the trained model instance.

## Loading and Making Predictions with the Model

To load the model back into memory, following BentoML's `load_model` function is used:

```python
import bentoml

iris_clf_runner = bentoml.sklearn.get("iris_svm_clf:latest").to_runner()

iris_clf_runner.init_local()

print(iris_clf_runner.predict.run([[5.9, 3.0, 5.1, 1.8]]))
```

You can then use this model for predicting unseen examples.

## Deploying the Model with BentoML

BentoML is a fantastic library that simplifies the process of serving and deploying ML models. One of its features is the ability to automatically generate a Swagger API from your PyTorch model.

By using BentoML, you can easily serve your model as a high-performance API endpoint and consume it from a web frontend, mobile applications or in a microservices architecture.

## Conclusion

This project is a demonstration of how one can build and deploy models with BentoML. You can access the complete code in this repository.

If you find it helpful, feel free to clone, download, or contribute to this project.

```
Remember to replace the dummy project links and contact details with yours before publishing. I hope this helps! Let me know if there's anything more I can assist you with.  
```