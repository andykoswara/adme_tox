# adme_tox
Python script that implements a random forest algorithm to predict several ADME/Tox classifications of bioactive molecules accompanied with a visualization technique called uniform manifold approximation projection (UMAP). This work is an amalgamation of a great many previous work by fellow researchers [ref] with an extension towards our own research work on predicting ion fragmentation by a mass spectrometer (MS).

'''ruby
import pandas as pd
import numpy as np
from adme_utils import *

from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, Descriptors

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
'''
