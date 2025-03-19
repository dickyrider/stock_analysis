import numpy as np

def ctl_label(df, threshold=0.05, delay=1):
    prices = df['Close'].values
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
            current_trend = 0
            current_extreme = prices[i]
            trend_start = i
            break

    # If the threshold is never reached, mark all as 0
    if current_trend is None:
        return labels

    # Set the labels before the trend_start to 0 (or handle as needed)
    labels[:trend_start] = -1
    # Mark the first confirmed trend point
    labels[trend_start] = current_trend

    # Check daily after the trend is established and update labels accordingly
    for i in range(trend_start + 1, n):
        price = prices[i]
        if current_trend == 1:  # Currently in an uptrend
            # Update the extreme if a new high is reached (applying the upward threshold)
            if price > current_extreme * (1 + threshold):
                current_extreme = price

            # If the price falls below the current extreme by the threshold, switch to downtrend
            if price < current_extreme * (1 - threshold):
                current_trend = 0
                current_extreme = price

        elif current_trend == 0:  # Currently in a downtrend
            # Update the extreme if a new low is reached (applying the downward threshold)
            if price < current_extreme * (1 - threshold):
                current_extreme = price

            # If the price rises above the current extreme by the threshold, switch to uptrend
            if price > current_extreme * (1 + threshold):
                current_trend = 1
                current_extreme = price
        labels[i] = current_trend
        
    df['Trend Label'] = labels

    # Shift the trend labels one day forward (to align next day’s label with current day’s features)
    df['Shifted Trend Label'] = df['Trend Label'].shift(-delay)
    
    return df

