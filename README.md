

<h3 align="center">AutoML Alex</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
[![Python](https://img.shields.io/badge/python-v3.7-blue.svg)]()

</div>

---

<p align="center"> State-of-the art Automated Machine Learning python library
    <br> 
</p>


## Installation

```python
pip install automl-alex
```


## 🚀 Examples

```python
from automl_alex import AutoMLClassifier

model = AutoMLClassifier(X_train, y_train, X_test,)
predict_test, predict_train = model.fit_predict(timeout=1000,)
```
More examples in the folder ./examples


## Features

- Data preprocessing
- Categorical feature Encoding
- Target Encoding with cross validation
- Cross Validation
- Search for the best solving library 
- Smart Optimization of Hyperparameters
- Timelimit and EarlyStoping
- Stacking

