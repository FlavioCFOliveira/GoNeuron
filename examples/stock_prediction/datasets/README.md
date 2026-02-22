# Stock Prediction Datasets

This directory contains historical stock price datasets from the [Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) repository.

## Dataset Structure

All CSV files follow the same format with the following columns:

| Column | Description |
|--------|-------------|
| Date | Trading date (YYYY-MM-DD) |
| Open | Opening price |
| High | Highest price during the day |
| Low | Lowest price during the day |
| Close | Closing price |
| Adj Close | Adjusted closing price (dividends, splits) |
| Volume | Number of shares traded |

## Available Datasets

| File | Company | Date Range | Records | Notes |
|------|---------|------------|---------|-------|
| FB.csv | Meta (Facebook) | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| TSLA.csv | Tesla | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| AMD.csv | AMD | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| TWTR.csv | Twitter | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| TMUS.csv | T-Mobile | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| GOOG.csv | Google (Alphabet) | 2017-10-02 to 2018-10-01 | 251 | Daily prices |
| KNX.csv | KNX | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| SINA.csv | Sina | 2018-05-23 to 2019-05-23 | 251 | Daily prices |
| INFY.csv |Infosys | 2018-05-23 to 2019-05-23 | 251 | Daily prices |

## Source

Datasets downloaded from: https://github.com/huseinzol05/Stock-Prediction-Models

## Usage for GoNeuron LSTM

For LSTM training in GoNeuron, you'll typically use the **Close** price as the target variable. The data can be normalized and split into sequences for time-series prediction.

Example preprocessing:
1. Normalize prices (e.g., Min-Max scaling or Z-score)
2. Create sequences of past N days to predict the next day's price
3. Split into train/test sets (e.g., 80/20)
