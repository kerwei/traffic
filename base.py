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
filename = 'selected_training.csv'
predict_fname = 'predict.csv'


def load_dataset(filename='training.csv'):
    df = pd.read_csv(os.path.join(DATADIR, filename))

    return df

"""
--- VALIDATION ---

Check that data input is consistent with downstream dataframe treatments. Reason: GIGO
"""
def validate_frame(df):
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
    # in a big list of errors, which is not useful when printed out to the console. For now, let's just raise a warning and continue
    errors = schema.validate(df)

    return errors

"""
--- DATETIME ---

Given that this is a time-series dataset, it is more convenient to work with datetime objects.
This section is aimed at creating corresponding datetime objects from the day and timestamp scalar values.
An arbritary calendar day 1-1-1900 will be selected as the base date. This can be updated later on if necessary
"""
BASEDATE = datetime(1900, 1, 1)
length = timedelta(minutes=15)
timemap = {(datetime(1900,1,1,0,0,0) + i * length).time(): i for i in range(96)}

def hrmin_scalar2delta(strseries):
    """
    Convert a time string with format hh:mm into its corresponding timedelta unit
    Base reference time is 00:00
    """
    for ts in strseries:
        x, y = ts.split(':')
        yield timedelta(hours=int(x), minutes=int(y))


def framebygeohash(df, geohashlist):
    """
    Filter the frame for the desired geohash
    """
    df_cut = pd.DataFrame(columns=df.columns)
    for ghash in geohashlist:
        this_cut = df.loc[df.geohash6 == ghash]
        df_cut = pd.concat([df_cut, this_cut])

    return df_cut


def standard_frame(frame):
    """
    Base frame operations
    """
    # Timedelta at hour and minute precision
    hrmin_delta = []
    hrmin_delta.extend(hrmin_scalar2delta(frame.timestamp))
    # Timedelta at day precision. -1 for 1900-01-01
    day_delta = [timedelta(days=x-1) for x in frame.day]
    # Timedelta combined
    time_delta = [x + y for x, y in zip(day_delta, hrmin_delta)]
    # Create the datetime column
    frame['reftime'] = [BASEDATE + x for x in time_delta]
    frame.set_index('reftime', inplace=True)

    return frame


def derived_frame(frame):
    """
    Create derived columns
    """
    # Populate time buckets with zero demand
    frame = frame.resample('15T').mean()
    frame.demand.fillna(0, inplace=True)
    # Day column needed to generate label and demand sequences
    frame['day'] = frame.index.dayofyear
    # frame['dmd_label'] = [''.join(['D', str(x//0.1)]) for x in frame.demand]
    # First label dimension: Demand is considered high (H) if it is above the median value and low (L), otherwise
    frame['dmd_label'] = ['H' if x > frame.demand.median() else 'L' for x in frame.demand]
    # Second label dimension: The ordinal value of a time bucket (T=15) starting from 12:00am
    frame['timex'] = frame.index.time
    frame['time_rank'] = frame.timex.map(timemap)
    # The label with the two dimensions combined
    frame['label'] = [''.join(['T', str(x), y]) for x, y in zip(frame.time_rank, frame.dmd_label)]
    # Tweak to allow HMM to do a one forward bucket prediction
    # For training, the emitted value is taken to be the value of the next bucket
    frame['rdemand'] = [str(round(x, 2)) for x in frame.demand]
    frame['forward_label'] = frame.label.shift(-1)
    # Drop unneeded columns
    frame.drop(['dmd_label', 'timex', 'time_rank'], axis=1, inplace=True)

    return frame


def extract():
    """
    Utility function to generate a mock dataset to be predicted
    """
    import random

    df = load_dataset()
    df = standard_frame(df)
    filename = 'data/predict.csv'

    geofilter = []
    for i in range(5):
        geofilter.append(random.choice(df.geohash6.unique()))

    geofilter = ['qp03pr', 'qp03x8', 'qp098h']
    filtered = df.loc[(df.geohash6 == geofilter[0]) & (df.index.dayofyear == 55) & (time(7, 0, 0) < df.index.time) & (df.index.time < time(9, 0, 0))]
    for g in geofilter[1:]:
        cut = df.loc[(df.geohash6 == g) & (df.index.dayofyear == 55) & (time(7, 0, 0) < df.index.time) & (df.index.time < time(9, 0, 0))]
        filtered = pd.concat([filtered, cut])

    filtered.reset_index(inplace=True)
    # filtered.demand = np.nan
    filtered.to_csv(filename)
        

if __name__ == '__main__':
    extract()
    # Load the default training set if no filenames are supplied
    df = load_dataset()
    # Run data validation on the frame
    errors = validate_frame(df)

    if errors:
        warnings.warn("Data loaded with some inconsistencies. Resulting dataframe may not be accurate.")

    # Load dataset to be predicted
    pset = load_dataset(predict_fname)
    geofilter = list(set([x for x in pset.geohash6]))
    pset = standard_frame(pset)

    # Only filter the relevant geohashes for training
    df = framebygeohash(df, geofilter)
    # Convert scalar day-time values to datetime objects
    df = standard_frame(df)

    # Perform the analysis geohash by geohash
    for geo in geofilter:
        df_geo = df.loc[df.geohash6 == geo]
        df_geo = derived_frame(df_geo)
        demand_label = list(df_geo.label.unique())

        # Train the emission model
        label_train = [list(x[1]) for x in df_geo.groupby(df_geo.index.dayofyear)['label']]
        rdemand_train = [list(x[1]) for x in df_geo.groupby(df_geo.index.dayofyear)['rdemand']]
        em_model = emission.bake_model(rdemand_train, label_train)

        # Train the transmission model for T+5 periods ahead
        # The shifting means that the last row has no forward label
        df_geo.drop(df_geo.loc[pd.isnull(df_geo.forward_label)].index, inplace=True)
        label_train = [list(x[1]) for x in df_geo.groupby(df_geo.index.dayofyear)['label']]
        forward_train = [list(x[1]) for x in df_geo.groupby(df_geo.index.dayofyear)['forward_label']]
        forward_model = emission.bake_model(forward_train, label_train)

        # Generate the transmission for the forward periods
        pset_geo = pset.loc[pset.geohash6 == geo]
        pset_geo = derived_frame(pset_geo)
        pset_labels = pset_geo.label.to_list()

        for i in range(5):
            res = utils.simplify_decoding(pset_labels, forward_model, demand_label)
            pset_labels.append(res[-1])

        # Extract the actual demands for the predicted period
        df_actual = df.loc[(df.geohash6 == geo) & (df.index.dayofyear == 55) & (time(8, 45, 0) < df.index.time) & (df.index.time < time(10, 15, 0))]

        # Prediction for the next T+5 (15-min buckets)
        pperiod = [(pd.to_datetime(pset_geo.iloc[-1].name) + i * timedelta(minutes=15)).time() for i in range(1,6)]
        plabels = pset_labels[-5:]

        print("Geohash: {}\n".format(geo))
        print("Lead predict label:\n--------------")
        print(pset_geo)
        print("\n\nPredicted transition:\n-----------------")
        print([(x,y) for x,y in zip(pperiod, plabels)])
        print("\n\nPredicted demand:\n-----------------")
        p_demand = utils.simplify_decoding(plabels, em_model, demand_label)
        df_predict = pd.DataFrame({
            'period': [str(x) for x in pperiod],
            'demand': [float(x) for x in p_demand]})
        print(df_predict)
        print("\n\nActual demand:\n--------------")
        sr_period = pd.Series([str(x) for x in df_actual.index.time], name='period')
        sr_period.index = df_actual.index
        df_actual = df_actual.assign(period=sr_period)
        df_actual = df_actual.loc[:,['period', 'demand']].sort_values(by='reftime')
        print(df_actual)
        print("\n\nDeviation:\n--------------")
        # df_actual.reset_index(inplace=True)
        # df_actual.drop('reftime', axis=1, inplace=True)
        df_actual.set_index('period', inplace=True)
        df_predict.set_index('period', inplace=True)
        df_error = df_predict.join(df_actual, on='period', lsuffix='_predict', rsuffix='_actual')
        # df_error = df_error.assign(sq_error=lambda x: (x.demand_predict - x.demand_actual) ** 2)
        print(df_error)

