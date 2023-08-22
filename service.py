import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_svm_clf").to_runner()

svc = bentoml.Service("iris-classifier", runners=[iris_clf_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    return iris_clf_runner.predict.run(input_series)
