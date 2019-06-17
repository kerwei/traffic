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
The Hidden Markov Model (HMM), more commonly used for the part of speech tagging will be adapted for use in this project. For this project, the demand level at each time bucket for a given geohash is labeled 'H' if it is above the median value and 'L' otherwise. Each time bucket is also numbered from 12:00, T0.0 to 23:45 T95.0. Together with the label for the level of demand, a combination label (tags) is constructed such as 'T55.0L'.

In part of speech tagging projects, the sequence of tags, together with their respective emission probabilities are combined to determine the likeliest tag that emits the observed word. For this project, the observed demand at each time is thought to be emitted by a combined time-demand tag. Therefore, to generate predictions for up to T + 5 periods ahead, the time-demand tags will be predicted first. Again, the HMM method is adapted to achieve this purpose by establishing that, for a given time-demand tag, the next-period time-demand tag is emitted with a calculated probability. For the training of the model, a 'forward_label' column is introduced, which is basically the combined time-demand label, shifted by 1 row. For example, for the label 'T55.0L', the next period demand level is the expected emission (either 'T56.0L' or 'T56.0H'). Once, this is performed iteratively for 5 periods ahead, the emitted time-demand tags then produce the expected demand, for each time bucket up to T+5. 

## Instructions
1. Save the training dataset as 'training.csv' inside the data folder `data/`
2. Save the set of data where predictions need to be made on as 'predict.csv' within the same data folder `data/`. The following columns should be present in order for a prediction to be made:
    a. geohash6
    b. day
    c. timestamp
3. Execute `python base.py` from the root directory on the console and the output will be printed to the console.
