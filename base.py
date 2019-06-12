from collections import Counter
from copy import copy
from datetime import datetime, timedelta, time
import geohash2
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas_schema import Column, Schema
from pandas_schema.validation import InRangeValidation, MatchesPatternValidation, IsDtypeValidation, InListValidation
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
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


def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    tagset = set(k for i in sequences_A for k in i)
    res_dct = {i:{} for i in tagset}
    cntr = Counter(zip(chain(*sequences_A), chain(*sequences_B)))
    for tag, word in cntr:
        res_dct[tag][word] = cntr[(tag, word)]    
    return res_dct


def replace_unknown(sequence, demand_label):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in demand_label else 'nan' for w in sequence]


def simplify_decoding(X, model, demand_label):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X, demand_label))
    return [state[1].name for state in state_path[1:-1]]  # do not show the start/end state predictions


# Timedelta at hour and minute precision
hrmin_delta = []
hrmin_delta.extend(hrmin_scalar2delta(df.timestamp))
# Timedelta at day precision. -1 for 1900-01-01
day_delta = [timedelta(days=x-1) for x in df.day]
# Timedelta combined
time_delta = [x + y for x, y in zip(day_delta, hrmin_delta)]
# Create the datetime column
df['reftime'] = [BASEDATE + x for x in time_delta]


"""
qp03mf
"""
# Cut the dataset up into training set and testing set
qp03mf = df.loc[df.geohash6 == 'qp03mf']
qp03mf.set_index(qp03mf.reftime, inplace=True)
qp03mf_quart = qp03mf.resample('15T').mean()
# Fill np.nan demands with 0s
qp03mf_quart.demand.fillna(0, inplace=True)
# Patch np.nan days
qp03mf_quart.day = qp03mf_quart.index.dayofyear

# Apply the labels - resolution of 0.1 normalized demand
# Resolution at 0.1 normalized demand
qp03mf_quart['dmd_label'] = [''.join(['D', str(x//0.1)]) for x in qp03mf_quart.demand]
# Resolution at 0.5 normalized demand
qp03mf_quart['dmd_label'] = ['H' if x >= qp03mf_quart.demand.median() else 'L' for x in qp03mf_quart.demand]
# Let's allow 15-min demand forecast to be made as long as we know the demand pattern for the past hour
# Therefore, tags should be the concatenation of demand for [T-4, T-3, T-2, T-1, T] -> 100,000 possibilities
# Should I reduce the resolution of these buckets??
qp03mf_quart['timex'] = qp03mf_quart.index.time
qp03mf_quart['time_rank'] = qp03mf_quart.timex.rank(method='dense')
qp03mf_quart['label'] = [''.join(['T', str(x), y]) for x, y in zip(qp03mf_quart.time_rank, qp03mf_quart.dmd_label)]
qp03mf_quart.drop(['dmd_label', 'timex', 'time_rank'], axis=1, inplace=True)
qp03mf_quart['rdemand'] = [str(round(x, 2)) for x in qp03mf_quart.demand]

# Split into training and testing sets
demand_label = list(qp03mf_quart.label.unique())
df_train = qp03mf_quart.loc[qp03mf_quart.day <= 48, ['label', 'rdemand']]
df_test = qp03mf_quart.loc[qp03mf_quart.day > 48, ['label', 'rdemand']]


"""
--- HMM TAGGER ---
"""
def unigram_counts(sequence):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.
    
    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    return Counter(sequence)

def bigram_counts(sequence):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
    pairlst = list(zip(sequence[:-1], sequence[1:]))
    
    return Counter(pairlst)

def starting_counts(sequence):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.
    
    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    return Counter(sequence)

def ending_counts(sequence):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.
    
    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """
    return Counter(sequence)


tag_unigrams = unigram_counts(df_train.rdemand)
tag_bigrams = bigram_counts(df_train.rdemand)

# Uniform distribution for starting and ending labels
all_labels = [str(x) for x in df_train.rdemand.unique()]
tag_starts = starting_counts(all_labels)
tag_ends = ending_counts(all_labels)

basic_model = HiddenMarkovModel(name="base-hmm-tagger")

# Emission count
label_train = [list(x[1]) for x in df_train.groupby(df_train.index.dayofyear)['label']]
rdemand_train = [list(x[1]) for x in df_train.groupby(df_train.index.dayofyear)['rdemand']]
emission_count = pair_counts(rdemand_train, label_train)

# States with emission probability distributions P(word | tag)
states = []
for rdemand, label_dict in emission_count.items() :
    dist_tag = DiscreteDistribution({label: cn/tag_unigrams[rdemand] for label, cn in label_dict.items()})
    states.append(State(dist_tag, name=rdemand))

basic_model.add_states(states)
state_names = [s.name for s in states]
state_index = {tag:num for num, tag in enumerate(state_names)}

# Start transition
total_start = sum(tag_starts.values())
# pdb.set_trace()
for tag, cn in tag_starts.items():
    sname = state_index[tag]
    basic_model.add_transition(basic_model.start, states[state_index[tag]], cn/total_start)

# End transition
total_end = sum(tag_ends.values())
for tag, cn in tag_ends.items():
    basic_model.add_transition(states[state_index[tag]], basic_model.end, cn/total_end)


# Edges between states for the observed transition frequencies P(tag_i | tag_i-1)
for key, value in tag_bigrams.items():
    basic_model.add_transition(states[state_index[key[0]]], states[state_index[key[1]]], value/tag_unigrams[key[0]])

# Finalize the model
basic_model.bake()


if __name__ == '__main__':
    # Testing set
    test = df_test.loc[(df_test.index.day == 1) & (time(9, 0, 0) < df_test.index.time) & (df_test.index.time < time(12, 0, 0)), 'label']
    actual = df_test.loc[(df_test.index.day == 1) & (time(9, 0, 0) < df_test.index.time) & (df_test.index.time < time(12, 0, 0)), 'rdemand']

    print("Sentence Key: {}\n".format(test))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(test, basic_model, demand_label))
    print()
    print("Actual labels:\n--------------")
    print(actual)
    print("\n")