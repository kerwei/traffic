from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from utils import *
from itertools import chain


def bake_model(tags_sequence, words_sequence):
    """
    'tags' are the time-demand labels that generate the emitted demand level.
    Demand level are represented by 'words'
    """
    # rdemand
    words = [x for x in chain(*words_sequence)]
    tag_unigrams = unigram_counts(words)
    tag_bigrams = bigram_counts(words)

    # Uniform distribution for starting and ending labels
    all_labels = list(set(words))
    tag_starts = starting_counts(all_labels)
    tag_ends = ending_counts(all_labels)

    basic_model = HiddenMarkovModel(name="base-hmm-tagger")

    # Emission count
    label_train = tags_sequence
    rdemand_train = words_sequence
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
    for tag, cn in tag_starts.items():
        # sname = state_index[tag]
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

    return basic_model