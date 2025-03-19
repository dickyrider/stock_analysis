import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt



# Download NVDA data from Yahoo Finance
nvda = yf.download("NVDA", start=start_date, end=end_date)
nvda.columns = [col[0] for col in nvda.columns]
nvda.reset_index(inplace=True)

def label_trends(prices, threshold=0.05):
    n = len(prices)
    labels = np.zeros(n, dtype=int)
    
    # Set the initial price and initialize variables
    initial_price = prices[0]
    current_trend = None   # Trend not determined yet; initialize to None
    current_extreme = initial_price
    trend_start = 0        # Index when the trend is confirmed

    # Determine the first trend turning point
    for i in range(1, n):
        if prices[i] >= initial_price * (1 + threshold):
            current_trend = 1
            current_extreme = prices[i]
            trend_start = i
            break
        elif prices[i] <= initial_price * (1 - threshold):
            current_trend = -1
            current_extreme = prices[i]
            trend_start = i
            break

    # If the threshold is never reached, mark all as 0
    if current_trend is None:
        return labels

    # Set the labels before the trend_start to 0 (or handle as needed)
    labels[:trend_start] = 0
    # Mark the first confirmed trend point
    labels[trend_start] = current_trend

    # Check daily after the trend is established and update labels accordingly
    for i in range(trend_start + 1, n):
        price = prices[i]
        if current_trend == 1:  # Currently in an uptrend
            # Update the extreme if a new high is reached (applying the upward threshold)
            if price > current_extreme * (1 + threshold):
                current_extreme = price
                labels[i] = current_trend
            # If the price falls below the current extreme by the threshold, switch to downtrend
            if price < current_extreme * (1 - threshold):
                current_trend = -1
                current_extreme = price
                labels[i] = current_trend
        elif current_trend == -1:  # Currently in a downtrend
            # Update the extreme if a new low is reached (applying the downward threshold)
            if price < current_extreme * (1 - threshold):
                current_extreme = price
                labels[i] = current_trend
            # If the price rises above the current extreme by the threshold, switch to uptrend
            if price > current_extreme * (1 + threshold):
                current_trend = 1
                current_extreme = price
                labels[i] = current_trend
        
    return labels

# Extract the closing prices and generate labels
prices = nvda['Close'].values
labels = label_trends(prices)

# Add the trend labels to the DataFrame
nvda['Trend Label'] = labels

# Shift the trend labels one day forward (to align next day’s label with current day’s features)
nvda['Shifted Trend Label'] = nvda['Trend Label'].shift(-1)

# Output the last 20 rows; note that the final row is NaN due to the shift
print(nvda[['Date', 'Close', 'Trend Label', 'Shifted Trend Label']].tail(20))

# Visualization: plot the closing price along with the original and shifted trend labels
plt.figure(figsize=(12,6))
plt.plot(nvda['Date'], nvda['Close'], label='Close Price', color='blue')

# Plot the shifted trend labels (only show rows with a non-NaN shifted label)
shifted = nvda.dropna(subset=['Shifted Trend Label'])
up_shifted = shifted[shifted['Shifted Trend Label'] == 1]
down_shifted = shifted[shifted['Shifted Trend Label'] == -1]
plt.scatter(up_shifted['Date'], up_shifted['Close'], label='Up Trend (Shifted)', facecolors='none', edgecolors='green', marker='^', s=100)
plt.scatter(down_shifted['Date'], down_shifted['Close'], label='Down Trend (Shifted)', facecolors='none', edgecolors='red', marker='v', s=100)

plt.title(f'{ticker} Close Price with Original and Forward Shifted Trend Labels')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
