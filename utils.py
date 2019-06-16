from collections import Counter
from itertools import chain


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