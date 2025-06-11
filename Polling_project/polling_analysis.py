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
party_columns = ['Lab', 'Con', 'Ref', 'LD', 'Greens', 'SNP', 'Others']

# Replace empty strings and whitespace with NaN
for col in party_columns:
    df[col] = pd.to_numeric(df[col].replace(r'^\s*$', np.nan, regex=True), errors='coerce')

# Calculate daily averages, excluding NaN values
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
    'SNP': '#000000',   # SNP Black
    'Others': '#999999'  # Grey
}

# Plot each party with LOESS smoothing
for party, color in colors.items():
    # Filter out NaN values for this party
    valid_data = daily_averages.dropna(subset=[party])
    
    if len(valid_data) > 0:
        # Create scatter plot with higher alpha for individual points
        plt.scatter(valid_data['Date'], valid_data[party], 
                    color=color, alpha=0.5, label=f'{party} (raw)')
        
        # Calculate LOESS smoothing with 14-day window
        # Convert 14 days to a fraction of the total date range
        date_range = (valid_data['Date'].max() - valid_data['Date'].min()).days
        frac = 14 / date_range  # 14 days as a fraction of total days
        
        lowess = sm.nonparametric.lowess(valid_data[party], 
                                        valid_data['Date_Numeric'],
                                        frac=frac, return_sorted=False)  # 14-day window
        
        # Plot smoothed line
        plt.plot(valid_data['Date'], lowess, 
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
# Calculate averages directly from the raw data, excluding NaN values
averages = df[party_columns].mean().round(1).sort_values(ascending=False)
latest_date = daily_averages['Date'].max()
latest_polls = daily_averages[daily_averages['Date'] == latest_date]

# Get the most recent poll details
most_recent_poll = df.sort_values('Date', ascending=False).iloc[0]
most_recent_date = most_recent_poll['Date']

# Create data dictionary for JSON
polling_data = {
    'most_recent_poll': {
        'date': most_recent_date.strftime('%d %B %Y'),
        'pollster': most_recent_poll['Pollster'],
        'sample_size': int(most_recent_poll['Sample Size']),
        'labour': float(most_recent_poll['Lab']) if pd.notna(most_recent_poll['Lab']) else 'Not Supplied',
        'conservative': float(most_recent_poll['Con']) if pd.notna(most_recent_poll['Con']) else 'Not Supplied',
        'reform': float(most_recent_poll['Ref']) if pd.notna(most_recent_poll['Ref']) else 'Not Supplied',
        'libdem': float(most_recent_poll['LD']) if pd.notna(most_recent_poll['LD']) else 'Not Supplied',
        'greens': float(most_recent_poll['Greens']) if pd.notna(most_recent_poll['Greens']) else 'Not Supplied',
        'snp': float(most_recent_poll['SNP']) if pd.notna(most_recent_poll['SNP']) else 'Not Supplied',
        'others': float(most_recent_poll['Others']) if pd.notna(most_recent_poll['Others']) else 'Not Supplied'
    },
    'latest_polls': {party: round(latest_polls[party].values[0], 1) if pd.notna(latest_polls[party].values[0]) else 'Not Supplied' for party in party_columns},
    'averages': {party: round(averages[party], 1) if pd.notna(averages[party]) else 'Not Supplied' for party in party_columns},
    'recent_polls': []
}

# Add recent polls data
recent_polls = df.sort_values('Date', ascending=False).head(10)
for _, poll in recent_polls.iterrows():
    polling_data['recent_polls'].append({
        'date': poll['Date'].strftime('%d %B %Y'),
        'pollster': poll['Pollster'],
        'sample_size': int(poll['Sample Size']),
        'labour': float(poll['Lab']) if pd.notna(poll['Lab']) else 'Not Supplied',
        'conservative': float(poll['Con']) if pd.notna(poll['Con']) else 'Not Supplied',
        'reform': float(poll['Ref']) if pd.notna(poll['Ref']) else 'Not Supplied',
        'libdem': float(poll['LD']) if pd.notna(poll['LD']) else 'Not Supplied',
        'greens': float(poll['Greens']) if pd.notna(poll['Greens']) else 'Not Supplied',
        'snp': float(poll['SNP']) if pd.notna(poll['SNP']) else 'Not Supplied',
        'others': float(poll['Others']) if pd.notna(poll['Others']) else 'Not Supplied'
    })

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
    if pd.notna(value):
        print(f"{party}: {value:.1f}%") 