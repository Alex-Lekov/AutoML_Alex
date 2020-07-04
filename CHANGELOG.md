# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.07.04]

### Changed
- remove target encoding
- remove norm data
- rebuild cross_val
- preparation for the addition of FEs

## [0.06.14]

### Changed
- add Docs in functions
- auto metric directions

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