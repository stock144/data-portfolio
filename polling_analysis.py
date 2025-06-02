import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from datetime import datetime, timedelta

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

# Calculate 7-day window size as a fraction of total days
total_days = (daily_averages['Date'].max() - daily_averages['Date'].min()).days
window_fraction = 14 / total_days  # Convert 14 days to a fraction of total time period

# Plot each party with LOESS smoothing
for party, color in colors.items():
    # Create scatter plot with low alpha for individual points
    plt.scatter(daily_averages['Date'], daily_averages[party], 
                color=color, alpha=0.3, label=f'{party} (raw)')
    
    # Calculate LOESS smoothing with 7-day window
    lowess = sm.nonparametric.lowess(daily_averages[party], 
                                    daily_averages['Date_Numeric'],
                                    frac=window_fraction)  # 7-day window
    
    # Plot smoothed line
    plt.plot(daily_averages['Date'], lowess[:, 1], 
             color=color, linewidth=2, label=party)

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

# Print some basic statistics
averages = daily_averages[party_columns].mean().round(1).sort_values(ascending=False)

# Print the sorted averages
print("\nAverage support by party:")
print(averages)

print("\nLatest polling averages:")
latest_date = daily_averages['Date'].max()
latest_polls = daily_averages[daily_averages['Date'] == latest_date]
print(f"\nAs of {latest_date.strftime('%d %B %Y')}:")
for party in party_columns:
    print(f"{party}: {latest_polls[party].values[0]:.1f}%") 