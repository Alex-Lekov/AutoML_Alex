# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [1.02.15]

A big update that changes the logic of work
### NEW
- Now processing the dataset is separated from the model for ease of use when you want to process the dataset yourself
- Separate transform allows us to save and transfer processing to new data
### ADD
- Save & Load processing
- Save & Load model
- Reduce memory usage processing


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