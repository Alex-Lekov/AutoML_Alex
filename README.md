

<h3 align="center">AutoML Alex</h3>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/automl-alex)
![PyPI](https://img.shields.io/pypi/v/automl-alex)
[![CodeFactor](https://www.codefactor.io/repository/github/alex-lekov/automl_alex/badge)](https://www.codefactor.io/repository/github/alex-lekov/automl_alex)

</div>

---

<p align="center"> State-of-the art Automated Machine Learning python library for Tabular Data</p>


## Installation

```python
pip install automl-alex
```


## 🚀 Examples

Classifier:
```python
from automl_alex import AutoMLClassifier

model = AutoMLClassifier(X_train, y_train, X_test,)
predict_test, predict_train = model.fit_predict(timeout=2000,)
```

Regression:
```python
from automl_alex import AutoMLRegressor

model = AutoMLRegressor(X_train, y_train, X_test,)
predict_test, predict_train = model.fit_predict(timeout=2000,)
```

More examples in the folder ./examples:

- [01_Quick_Start.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/01_Quick_Start.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/01_Quick_Start.ipynb)
- [02_Models_v1.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/02_Models_v1.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/02_Models_v1.ipynb)
- [03_Data_Cleaning_and_Encoding_(DataBunch).ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/03_Data_Cleaning_and_Encoding_(DataBunch).ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/03_Data_Cleaning_and_Encoding_(DataBunch).ipynb)
- [04_ModelsReview.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/04_ModelsReview.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/04_ModelsReview.ipynb)
- [05_BestSingleModel.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/05_BestSingleModel.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/05_BestSingleModel.ipynb)


## Features

- Data preprocessing
- Categorical feature Encoding
- Target Encoding with cross validation
- Cross Validation
- Search for the best solving library 
- Smart Optimization of Hyperparameters (TPE)
- Timelimit and EarlyStoping
- Stacking

