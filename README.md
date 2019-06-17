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

## Implementation
The Hidden Markov Model, more commonly used for the part of speech tagging will be adapted for use in this project. The model is trained to recognized a time-demand level combination as a 'tag' and its corresponding demand at that point of time is considered to be the emitted, observable outcome. Given that the geography is also a big factor in predicting demands, the geohashes are mutually exclusive and therefore, for each of prediction needed to be made at a particular geohash, we will require the model to be trained on its historical data beforehand.

## Instructions
1. Save the training dataset as 'training.csv' inside the data folder `data/`
2. Save the set of data where predictions need to be made on as 'predict.csv' within the same data folder `data/`. The following columns should be present in order for a prediction to be made:
    a. geohash6
    b. day
    c. timestamp
3. Execute `python base.py` from the root directory on the console and the output will be printed to the console.
