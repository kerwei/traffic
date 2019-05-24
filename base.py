import os
import pandas as pd
from pandas_schema import Column, Schema
from pandas_schema.validation import InRangeValidation, MatchesPatternValidation, IsDtypeValidation, InListValidation
import numpy as np
import sys
import warnings

import pdb


ROOTDIR = os.getcwd()
DATADIR = os.path.join(ROOTDIR, 'data')
filename = 'training.csv'

df = pd.read_csv(os.path.join(DATADIR, filename))

### VALIDATION ###
schema = Schema([
    Column('geohash6', [MatchesPatternValidation(r'[a-z0-9]{6}')]),
    Column('day', [IsDtypeValidation(np.int64)]),
    Column('timestamp', [InListValidation(['20:0', '14:30', '6:15', '5:0', '4:0', '12:15', '3:30', '20:45',
       '22:15', '9:15', '11:45', '14:45', '2:30', '23:45', '11:30',
       '10:0', '11:0', '18:30', '6:0', '13:0', '4:30', '15:30', '4:15',
       '9:0', '0:15', '21:15', '4:45', '12:30', '12:0', '14:15', '9:30',
       '5:15', '3:15', '16:30', '8:0', '11:15', '18:45', '16:0', '2:15',
       '7:0', '18:0', '3:0', '15:0', '22:45', '20:30', '0:30', '13:30',
       '22:0', '5:30', '9:45', '10:30', '17:0', '5:45', '6:30', '23:30',
       '1:15', '0:45', '1:30', '13:45', '12:45', '2:45', '19:15', '14:0',
       '13:15', '15:45', '8:45', '23:15', '16:15', '19:30', '21:30',
       '10:45', '7:15', '7:30', '16:45', '17:15', '23:0', '6:45', '18:15',
       '1:0', '8:15', '17:45', '22:30', '2:0', '1:45', '7:45', '10:15',
       '3:45', '8:30', '15:15', '21:0', '21:45', '19:45', '19:0', '0:0',
       '17:30', '20:15'])]),
    Column('demand', [InRangeValidation(0, np.nextafter(1, 2))])
])

# pandas_schema does not currently support error summary. For validation of big datasets, this results
# in a big list of errors, which is not useful when printed out to the console. For now, let's just raise an warning and continue
errors = schema.validate(df)
if errors:
    warnings.warn("Data loaded with some inconsistencies. Resulting dataframe may not be accurate.")

