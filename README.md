

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
- [02_Models.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/02_Models.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/02_Models.ipynb)
- [03_Data_Cleaning_and_Encoding_(DataBunch).ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/03_Data_Cleaning_and_Encoding_(DataBunch).ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/03_Data_Cleaning_and_Encoding_(DataBunch).ipynb)
- [04_ModelsReview.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/04_ModelsReview.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/04_ModelsReview.ipynb)
- [05_BestSingleModel.ipynb](https://github.com/Alex-Lekov/AutoML_Alex/blob/master/examples/05_BestSingleModel.ipynb)  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/Alex-Lekov/AutoML_Alex/blob/master/examples/05_BestSingleModel.ipynb)


# What's inside

It integrates many popular frameworks:
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- ...


# Note:

- **With a large dataset, a lot of memory is required!**
Library creates many new features. If you have a large dataset with a large number of features (more than 100), you may need a lot of memory.
- **Do not work with timeseries and texts** yet


# Road Map

-   [x] Feature Generation

-   [ ] Advanced Logging

-   [ ] New Features Generators

-   [ ] DL Encoders

-   [ ] Save/Load and Predict on New Samples

-   [ ] Add More libs (NNs)

-   [ ] Add opt Pruners

-   [ ] Build pipelines

-   [ ] Docs Site


# Contact

[Telegram Group](https://t.me/automlalex)

