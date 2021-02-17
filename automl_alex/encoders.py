
import pandas as pd
import numpy as np
from category_encoders import HashingEncoder, SumEncoder, PolynomialEncoder, BackwardDifferenceEncoder 
from category_encoders import OneHotEncoder, HelmertEncoder, OrdinalEncoder, CountEncoder, BaseNEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, WOEEncoder, JamesSteinEncoder
from category_encoders.count import CountEncoder

# disable chained assignments
pd.options.mode.chained_assignment = None

################################################################
            #               Simple Encoders 
            #      (do not use information about target)
################################################################

cat_encoders_names = {
                'HashingEncoder': HashingEncoder,
                'SumEncoder': SumEncoder,
                'PolynomialEncoder': PolynomialEncoder,
                'BackwardDifferenceEncoder': BackwardDifferenceEncoder,
                'OneHotEncoder': OneHotEncoder,
                'HelmertEncoder': HelmertEncoder,
                'OrdinalEncoder': OrdinalEncoder,
                'CountEncoder': CountEncoder,
                'BaseNEncoder': BaseNEncoder,
                }

################################################################
            #                Target Encoders
################################################################

target_encoders_names = {
                'TargetEncoder': TargetEncoder,
                'CatBoostEncoder': CatBoostEncoder,
                'WOEEncoder': WOEEncoder,
                'JamesSteinEncoder': JamesSteinEncoder,
                }