# Grab Traffic Management Challenge

## Problem Statement
Economies in Southeast Asia are turning to AI to solve traffic congestion, which hinders mobility and economic growth. The first step in the push towards alleviating traffic congestion is to understand travel demand and travel patterns within the city. 

Can we accurately forecast travel demand based on historical Grab bookings to predict areas and times with high travel demand?

## Details
In this challenge, participants are to build a model trained on a historical demand dataset, that can forecast demand on a Hold-out test dataset. The model should be able to accurately forecast ahead by T+1 to T+5 time intervals (where each interval is 15-min) given all data up to time T.

The given dataset contains normalised (range[0,1]) historical demand of a city, aggregated spatiotemporally within geohashes and over 15 minute intervals. The dataset spans over a two month period. A brief description of the dataset fields are found below:

Field | Description
------------ | -------------
geohash6 | Geohash is a public domain geocoding system which encodes a geographic location into a short string of letters and digits with arbitrary precision. You may use the Python Geohash package https://pypi.org/project/Geohash/ or any Java Geohash library https://github.com/kungfoo/geohash-java or similar to encode and decode geohash into latitude and longitude and vice versa.
day | sequential order of days
timestsamp | start time of 15-minute intervals, in the following format: <hour>:<minute>, where hour ranges from 0 to 23 and minute is either one of (0, 15, 30, 45)
demand | aggregated demand normalised to be in the range [0,1]

## Methodology
### Background
The Hidden Markov Model (HMM), specifically the implementation used for part of speech tagging will be adapted for application in this project. In part of speech tagging, the HMM is used to determine the syntactic category (unobserved state) of a word (observed state), based on the words in its surrounding context. For example, the word _will_ in the sentecnce _"I will see Sally."_ on its own, can be a verb or a noun. A naive dictionary search would assign the unobserved state with either categories with a 50-50 probability. However, since the HMM also takes into account the composition of the sentence i.e. the surrounding words, it is able to correctly tag _will_ as a verb in this instance, rather than a noun.

### Viewing the challenge from a tagging perspective
To apply the same approach to the challenge of travel demand pattern, it is assumed that the demand at each point (unobserved state) in time emits a certain indicator state (observed state). To do this, the demand is first discretize by rounding the floating numbers down to 2 decimal points. Otherwise, there will be too many states in the ecosystem and the application of the model will be too costly. For a range of 0 to 1, this resulted in 101 possible states.

With the discrete demand states at hand, the observable indicator state is then generated through a simple rule - demand above the median value is assigned with a "High" tag and "Low", otherwise. Example:

Original demand | Rounded demand | Tag
--------------- | -------------- | ---
0.011234 | 0.01 | Low
0.031122 | 0.03 | High
0.0045 | 0.0 | Low
0.066778 | 0.07 | High
0.00891 | 0.01 | Low

### Enriching tags with additional information
It is probably commonly accepted that travel demand is driven by a multitude of factors. Therefore, it would be useful if these factors can be assimilated into the model as much as possible. Two other dimensions to the challenge are made available:
1. Geography
2. Time

For this project, the time aspect of the challenge is chosen as the "enrichment" property for observable tags. To make use of _time_, each of the time unit is assigned with an ordinal value from 0 at 12:00am to 95 at 11:45pm. 

Timestamp | Time ordinal value | Demand tag | Enriched tag
--------- | ------------------ | ---------- | ------------ 
09:00:00 | 36 | Low | T36L
09:15:00 | 37 | High | T37H
09:30:00 | 38 | Low | T38L
09:45:00 | 39 | High | T39H
10:00:00 | 40 | Low | T40L

### Determine travel demand from observable states
With the observable states now set up, the model is now ready to determine the likeliest travel demand (unobserved states) pattern that results in a certain observation. Example:

Observed state | Predicted demand | Original demand
-------------- | ---------------- | ---------------
T36L | 0.01 | 0.011234
T37H | 0.03 | 0.031122
T38L | 0.01 | 0.0045
T39H | 0.07 | 0.066778
T40L | 0.01 | 0.00891

### Generate forward observable states
So far, the project has been set up to generate the series of likeliest travel demand given a set of states. However, to be able to really anticipate the travel demand for a given period of time into the future, the model first requires the future observed states.

To do this by leveraging on the existing model, a second perspective needs to be added to the model. In the earlier section, the model assumes that each unobserved demand emits a certain observable demand level indicator. The same assumption shall be applied here. This time around, the future indicator tags are designated as the unobserved states, while the current-period indicator tag are the observable states.

Unobserved state | Observed state
---------------- | ---------------- 
T36L | T35L
T37H | T36L
T38L | T37H
T39H | T38L
T40L | T39H

With an existing series of observed demand level up to time T, we can then generate the likeliest demand level one time unit into the future. In the example shown above, given the observed states from T35 to T39, the likeliest series of unobserved states that resulted in the emission is shown to be 
> T36L - T37H - T38L - T39H - T40L

To generate a series of observable states up to T + 5, the same process is iterated 5 times, with the last term in the series of unobserved states added to the end of the series of observed states at each iteration.


## Instructions
1. Save the training dataset as 'training.csv' inside the data folder `data/`
2. Save the set of data where predictions need to be made on as 'predict.csv' within the same data folder `data/`. The following columns should be present in order for a prediction to be made:
    a. geohash6
    b. day
    c. timestamp
3. Execute `python base.py` from the root directory on the console and the output will be printed to the console.
