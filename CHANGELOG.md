# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [2023.3.11]
### Changed
- opt LVL


## [2023.3.9]
### Changed
- Update dependencies
### Fix
- ValueError: X and y both have indexes, but they do not match.


## [1.3.10]
### Fix
- TypeError in data_prepare Outliers filter


## [1.3.9]
### ADD
- Up score AutoML (Blend best top5 models in AutoML)


## [1.3.8]
### ADD
- optimization DataPreproc parametrs in BestSingleModel
- rebuild AutoML pepline  (light version)
### Fix
- target encodet only cat features


## [1.3.7]
### Fix
- target encoder in model.opt


## [1.3.6]
### ADD
- add dosc on CV


## [1.3.5]
### Fix
- Fix nans in targetencoder in CV


## [1.3.4]
### ADD
- Target Encoding in CrossValidation
- DenoisingAutoencoder in DataPrepare
- Docs


## [1.3.1]
### Fix
- Fix import - add loguru and psutil in requirements.txt


## [1.2.28]
### ADD
- Advanced Logging (logs in .automl-alex_tmp/log.log)
- Class Optimizer
- Pruner in optimizer
- connection with [optuna-dashboard](https://github.com/optuna/optuna-dashboard) (run > optuna-dashboard sqlite:///db.sqlite3 )
- NumericInteractionFeatures Class in data_prepare


## [1.2.25]
### Fix
- Fix save & load in AutoML

### ADD
- Metod .score() and .fit_score() in Models
- Class CrossValidation() examples in ./examples/03_Models.ipynb


## [1.2.24]
### Fix
- same Fixses in AutoML
### ADD
- New info in Readme.md


## [1.2.23]

A big update that changes the logic of work
### NEW
- Now processing the dataset is separated from the model for ease of use when you want to process the dataset yourself
- Separate transform allows us to save and transfer processing to new data
### ADD
- Save & Load processing
- Save & Load model
- Reduce memory usage processing
- Detect and remove outliers


## [1.01.11]

### Fix
- score_cv_folds fix in ModelsReview
- normalization


## [0.11.24]

### ADD
- multivariate TPE sampler. This algorithm captures dependencies among hyperparameters better than the previous algorithm

### Fix
- "ValueError non-broadcastable output operand..." in AutoMLRegressor


## [0.10.07]

### Fix
- DataConversionWarning in sklearn_models model.fit(X_train, y_train,)


## [0.10.04]

### Fix
- verbose in LinearRegression


## [0.08.05]

### Fix
- if y_train is not pd.DataFrame


## [0.07.26]

### Add
- Calc predict policy in AutoML

### Fix
- timelemit in AutoML (deleted Catboost in optimization)


## [0.07.25]

### Add
- Stacking in AutoML
- fit on full X_Train (no_CV)
- predict on full X in model_1 AutoML


## [0.07.21]

### Fixed
- AutoML model_2 score


## [0.07.20]

### Add
- Iterations in .opt

### Fixed
- timelemit in AutoML
- Num Features Generator in empty Num Features list


## [0.07.18]

### Add
- Features Generation in DataBunch
- Features Selection in .opt
- Generator interaction Num Features
- Generator FrequencyEncoder Features
- Generator Group Encoder Features
- Normalization Data
- Feature Importance


### Fixed
- RandomForest min_samples_split size
- fix ModelsReview opt cv


## [0.07.04]

### Changed
- remove target encoding
- remove norm data
- rebuild cross_val
- preparation for the addition of FEs


## [0.06.14]

### Changed
- add Docs in functions

### Fixed
- Try Fix .self buffer bug
- Fix dataset size < 1000


## [0.05.19]

### Changed
- Default stack_top=10 in AutoML


## [0.05.16]

### Changed
- predicts in DataFrame

### Added
- predicts from configs


## [0.05.11]

### Added
- RepeatedKFold in CV for prediction. 
- `n_repeats=2` in .cv()

### Changed
- Stacking metamodel now `LinearModel`
- in Stacking .predict `n_repeats=2` => `n_repeats=1` (timelimit :( )

### Fixed
- Fix Timelimit Error in Stacking