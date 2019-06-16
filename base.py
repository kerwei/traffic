from copy import copy
from datetime import datetime, timedelta, time
import numpy as np
import os
import pandas as pd
from pandas_schema import Column, Schema
from pandas_schema.validation import InRangeValidation, MatchesPatternValidation, IsDtypeValidation, InListValidation
import emission
import utils
import warnings

import pdb


ROOTDIR = os.getcwd()
DATADIR = os.path.join(ROOTDIR, 'data')
filename = 'training.csv'

df = pd.read_csv(os.path.join(DATADIR, filename))

"""
--- VALIDATION ---

Check that data input is consistent with downstream dataframe treatments. Reason: GIGO
"""
schema = Schema([
    Column('geohash6', [MatchesPatternValidation(r'[a-z0-9]{6}')]),
    Column('day', [IsDtypeValidation(np.int64), InRangeValidation(min=1)]),
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


"""
--- DATETIME ---

Given that this is a time-series dataset, it is more convenient to work with datetime objects.
This section is aimed at creating corresponding datetime objects from the day and timestamp scalar values.
An arbritary calendar day 1-1-1900 will be selected as the base date. This can be updated later on if necessary
"""
BASEDATE = datetime(1900, 1, 1)

def hrmin_scalar2delta(strseries):
    """
    Convert a time string with format hh:mm into its corresponding timedelta unit
    Base reference time is 00:00
    """
    for ts in strseries:
        x, y = ts.split(':')
        yield timedelta(hours=int(x), minutes=int(y))


def framebygeohash(df, geohashlist):
    # Cut the dataset up into training set and testing set
    df_cut = pd.DataFrame(columns=df.columns)
    for ghash in geohashlist:
        this_cut = df.loc[df.geohash6 == ghash]
        df_cut = pd.concat([df_cut, this_cut])

    return df_cut


# All geohashes
geohash = [x for x in df.geohash6]

geofilter = geohash[:10]
df = framebygeohash(df, geofilter)

# Timedelta at hour and minute precision
hrmin_delta = []
hrmin_delta.extend(hrmin_scalar2delta(df.timestamp))
# Timedelta at day precision. -1 for 1900-01-01
day_delta = [timedelta(days=x-1) for x in df.day]
# Timedelta combined
time_delta = [x + y for x, y in zip(day_delta, hrmin_delta)]
# Create the datetime column
df['reftime'] = [BASEDATE + x for x in time_delta]
df.set_index('reftime', inplace=True)
df = df.resample('15T').mean()
df.demand.fillna(0, inplace=True)
df['day'] = df.index.dayofyear
df['dmd_label'] = [''.join(['D', str(x//0.1)]) for x in df.demand]
df['dmd_label'] = ['H' if x >= df.demand.median() else 'L' for x in df.demand]
df['timex'] = df.index.time
df['time_rank'] = df.timex.rank(method='dense')
df['label'] = [''.join(['T', str(x), y]) for x, y in zip(df.time_rank, df.dmd_label)]
df.drop(['dmd_label', 'timex', 'time_rank'], axis=1, inplace=True)
df['rdemand'] = [str(round(x, 2)) for x in df.demand]
df['forward_label'] = df.label.shift(-1)


# Split into training and testing sets
demand_label = list(df.label.unique())
df_train = df.loc[df.day <= 48, ['label', 'rdemand', 'forward_label']]
df_test = df.loc[df.day > 48, ['label', 'rdemand', 'forward_label']]


if __name__ == '__main__':
    # TODO: Set up a loop for multiple geohashes
    # Right now I can't be sure of the result output
    # Testing set
    test = df_test.loc[(df_test.index.day == 1) & (time(9, 0, 0) < df_test.index.time) & (df_test.index.time < time(12, 0, 0)), 'label']
    actual = df_test.loc[(df_test.index.day == 1) & (time(9, 0, 0) < df_test.index.time) & (df_test.index.time < time(12, 0, 0)), 'rdemand']

    # Train the model
    label_train = [list(x[1]) for x in df_train.groupby(df_train.index.dayofyear)['label']]
    rdemand_train = [list(x[1]) for x in df_train.groupby(df_train.index.dayofyear)['rdemand']]
    em_model = emission.bake_model(label_train, rdemand_train)

    print("Sentence Key: {}\n".format(test))
    print("Predicted labels:\n-----------------")
    print(utils.simplify_decoding(test, em_model, demand_label))
    print()
    print("Actual labels:\n--------------")
    print(actual)
    print("\n")