from category_encoders import (
    HashingEncoder,
    SumEncoder,
    PolynomialEncoder,
    BackwardDifferenceEncoder,
)
from category_encoders import (
    OneHotEncoder,
    HelmertEncoder,
    OrdinalEncoder,
    BaseNEncoder,
)
from category_encoders.count import CountEncoder

################################################################
#               Simple Encoders
#      (do not use information about target)
################################################################

cat_encoders_names = {
    "HashingEncoder": HashingEncoder,
    "SumEncoder": SumEncoder,
    "BackwardDifferenceEncoder": BackwardDifferenceEncoder,
    "OneHotEncoder": OneHotEncoder,
    "HelmertEncoder": HelmertEncoder,
    "BaseNEncoder": BaseNEncoder,
    "CountEncoder": CountEncoder,
}
