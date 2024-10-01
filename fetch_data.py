import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split

def fetch_data():
    # Fetch 1-minute data for the last 5 days for Google (GOOGL)
    data = yf.download("GOOGL", period="5d", interval="1m")

    # Save the 'Close' column data
    df = data[['Close']].dropna() #For testing purposes, wouldn't usually do this.

    # Split the data into training and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

    # Save the data to CSV
    train_data.to_csv('train_data.csv', index=True)
    test_data.to_csv('test_data.csv', index=True)

    print("Minute-level data fetched and saved as train_data.csv and test_data.csv")

if __name__ == "__main__":
    fetch_data()
