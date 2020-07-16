#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os

def load_requirements(filename):
    with open(os.path.join('.', filename), "r") as f:
        return f.read().splitlines()

with open("automl_alex/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

import io
with io.open('README.md', encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='automl_alex',
    version=version,
    description='State-of-the art Automated Machine Learning python library for Tabular Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alex Lekov',
    author_email='itslek@yandex.ru',
    keywords=['machine learning', 'data science', 'automated machine learning', 'automl', 'hyperparameter optimization', 'artificial intelligence', 'ensembling', 'stacking', 'blending', 'deep learning', 'tensorflow', 'deeplearning', 'lightgbm', 'gradient boosting', 'gbm', 'keras', ],
    packages=['automl_alex', 'automl_alex.models'],
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        'License :: OSI Approved :: MIT License',
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6.*',
    install_requires=load_requirements("requirements.txt"),
    test_suite='nose.collector',
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/Alex-Lekov/AutoML_Alex/issues',
        'Source': 'https://github.com/Alex-Lekov/AutoML_Alex/',
    },
)