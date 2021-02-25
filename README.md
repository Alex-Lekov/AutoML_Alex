

<h3 align="center">AutoML Alex</h3>

<div align="center">

[![Downloads](https://pepy.tech/badge/automl-alex)](https://pepy.tech/project/automl-alex)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/automl-alex)
![PyPI](https://img.shields.io/pypi/v/automl-alex)
[![CodeFactor](https://www.codefactor.io/repository/github/alex-lekov/automl_alex/badge)](https://www.codefactor.io/repository/github/alex-lekov/automl_alex)
[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/automlalex)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> State-of-the art Automated Machine Learning python library for Tabular Data</p>

<img width=800 src="https://github.com/Alex-Lekov/AutoML-Benchmark/blob/master/img/Total_SUM.png" alt="bench">


From [AutoML-Benchmark](https://github.com/Alex-Lekov/AutoML-Benchmark/) 

### Scheme
<img width=800 src="https://github.com/Alex-Lekov/AutoML_Alex/blob/develop/examples/img/shema.png" alt="scheme">


# Features

- Automated Data Clean (Auto Clean)
- Automated **Feature Engineering** (Auto FE)
- Smart Hyperparameter Optimization (HPO)
- Feature Generation
- Feature Selection
- Models Selection
- Cross Validation
- Timelimit and EarlyStoping


# Installation

```python
pip install automl-alex
```


# 🚀 Examples

Classifier:
```python
from automl_alex import AutoMLClassifier

model = AutoMLClassifier()
model = model.fit(X_train, y_train, timeout=600)
predicts = model.predict(X_test)
```

Regression:
```python
from automl_alex import AutoMLRegressor

model = AutoMLRegressor()
model = model.fit(X_train, y_train, timeout=600)
predicts = model.predict(X_test)
```

DataPrepare:
```python
from automl_alex import DataPrepare

de = DataPrepare()
X_train = de.fit_transform(X_train)
X_test = de.transform(X_test)
```

More examples in the folder ./examples:

- [01_Quick_Start.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/01_Quick_Start.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/01_Quick_Start.ipynb)
- [02_Data_Cleaning_and_Encoding_(DataPrepare).ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/02_Data_Cleaning_and_Encoding_(DataPrepare).ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/02_Data_Cleaning_and_Encoding_(DataPrepare).ipynb)
- [03_Models.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/03_Models.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/03_Models.ipynb)
- [04_ModelsReview.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/04_ModelsReview.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/04_ModelsReview.ipynb)



# What's inside

It integrates many popular frameworks:
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- ...


# Works with Features

-   [x] Categorical Features

-   [x] Numerical Features

-   [x] Binary Features

-   [ ] Text

-   [ ] Datetime

-   [ ] Timeseries

-   [ ] Image


# Note

- **With a large dataset, a lot of memory is required!**
Library creates many new features. If you have a large dataset with a large number of features (more than 100), you may need a lot of memory.


# Road Map

-   [x] Feature Generation

-   [x] Save/Load and Predict on New Samples

-   [ ] Advanced Logging

-   [ ] DL Encoders

-   [ ] Add More libs (NNs)

-   [ ] Add opt Pruners

-   [ ] Build pipelines

-   [ ] Docs Site


# Contact

[Telegram Group](https://t.me/automlalex)

