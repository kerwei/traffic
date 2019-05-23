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

## Initial Thoughts
1. Convert day + timestamp to datetime objects
2. Align the beginning of training set days to actual days on the calendar to identify weekends and public holidays
3. _NOTE_ Aggregated data is normalized across all geohash islandwide, in buckets of 15 minutes. Therefore, a value of 1.0 does not correspond to the same scalar value between 2 time buckets