import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from datetime import datetime, timedelta
import json

# Set the style for seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [14, 8]

# Read the CSV file
df = pd.read_csv("polls.csv")

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Date_Numeric'] = (df['Date'] - df['Date'].min()).dt.days

# Group by date and calculate mean for each party
party_columns = ['Lab', 'Con', 'Ref', 'LD', 'Greens', 'Others']
daily_averages = df.groupby('Date')[party_columns].mean().reset_index()

# Sort by date
daily_averages['Date_Numeric'] = (daily_averages['Date'] - daily_averages['Date'].min()).dt.days

# Create the plot
plt.figure(figsize=(14, 8))

# Define party colors
colors = {
    'Lab': '#E4003B',  # Labour Red
    'Con': '#0087DC',  # Conservative Blue
    'Ref': '#12B6CF',  # Reform Blue
    'LD': '#FAA61A',   # Liberal Democrat Orange
    'Greens': '#6AB023', # Green Party Green
    'Others': '#999999'  # Grey
}

# Plot each party with LOESS smoothing
for party, color in colors.items():
    # Create scatter plot with higher alpha for individual points
    plt.scatter(daily_averages['Date'], daily_averages[party], 
                color=color, alpha=0.5, label=f'{party} (raw)')
    
    # Calculate LOESS smoothing with 14-day window
    # Convert 14 days to a fraction of the total date range
    date_range = (daily_averages['Date'].max() - daily_averages['Date'].min()).days
    frac = 14 / date_range  # 14 days as a fraction of total days
    
    # Sort data by date for LOESS
    sorted_data = daily_averages.sort_values('Date')
    lowess = sm.nonparametric.lowess(sorted_data[party], 
                                    sorted_data['Date_Numeric'],
                                    frac=frac)  # 14-day window
    # Convert lowess[:, 0] (numeric days) back to datetime
    min_date = sorted_data['Date'].min()
    lowess_dates = [min_date + pd.Timedelta(days=int(x)) for x in lowess[:, 0]]
    # Plot smoothed line
    plt.plot(lowess_dates, lowess[:, 1], 
             color=color, linewidth=2, label=party)

# Add vertical line for Local Elections
election_date = pd.to_datetime('01/05/2025', format='%d/%m/%Y')
plt.axvline(x=election_date, color='black', linestyle='--', alpha=0.5)
plt.text(election_date, 38, 'Local Elections - England', 
         rotation=45, verticalalignment='bottom', 
         horizontalalignment='left', fontsize=10)

# Customize the plot
plt.title('UK Polling Trends', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Support (%)', fontsize=12)
plt.ylim(0, 40)  # Set y-axis limits from 0 to 40

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)

# Customize legend
plt.legend(title='Party', 
          bbox_to_anchor=(1.05, 1),
          loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('polling_trends.png', dpi=300, bbox_inches='tight')

# Calculate statistics
# Calculate averages directly from the raw data, not from daily averages
averages = df[party_columns].mean().round(1).sort_values(ascending=False)
latest_date = daily_averages['Date'].max()
latest_polls = daily_averages[daily_averages['Date'] == latest_date]

# Get the most recent poll details
most_recent_poll = df.sort_values('Date', ascending=False).iloc[0]
most_recent_poll_info = {
    'date': most_recent_poll['Date'].strftime('%d %B %Y'),
    'pollster': most_recent_poll['Pollster']
}

# Create data dictionary for JSON
polling_data = {
    'averages': averages.to_dict(),
    'latest_date': latest_date.strftime('%d %B %Y'),
    'latest_polls': {party: round(latest_polls[party].values[0], 1) for party in party_columns},
    'most_recent_poll': most_recent_poll_info
}

# Save to JSON file
with open('polling_data.json', 'w') as f:
    json.dump(polling_data, f, indent=4)

# Print statistics
print("\nAverage support by party:")
print(averages)

print("\nLatest polling averages:")
print(f"\nAs of {latest_date.strftime('%d %B %Y')}:")
# Get latest polls and sort in descending order
latest_values = latest_polls[party_columns].iloc[0]
sorted_latest = latest_values.sort_values(ascending=False)
for party, value in sorted_latest.items():
    print(f"{party}: {value:.1f}%") 