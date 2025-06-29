import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

# Set the style for seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [14, 8]

# Read the seat results
df = pd.read_json("all_seat_results.json")

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
df = df.sort_values('date')

# Extract seat counts into columns
seat_parties = ['Labour', 'Conservative', 'Reform UK', 'Liberal Democrats', 'SNP', 'Others', 'Green', 'Plaid Cymru']
for party in seat_parties:
    df[party] = df['seats'].apply(lambda x: x.get(party, np.nan))

# Group by date and calculate mean seats for each party
daily_averages = df.groupby('date')[seat_parties].mean().reset_index()

# Plotting colors (same as polling_analysis.py)
colors = {
    'Labour': '#E4003B',
    'Conservative': '#0087DC',
    'Reform UK': '#12B6CF',
    'Liberal Democrats': '#FAA61A',
    'Green': '#6AB023',
    'SNP': '#000000',
    'Plaid Cymru': '#005B54',
    'Others': '#999999'
}

plt.figure(figsize=(14, 8))

for party, color in colors.items():
    valid_data = daily_averages.dropna(subset=[party])
    if len(valid_data) > 0:
        # Rolling mean smoothing only (no scatter)
        rolling = valid_data[party].rolling(window=7, min_periods=1, center=True).mean()
        plt.plot(valid_data['date'], rolling, color=color, linewidth=2, label=party)

# Add horizontal line for majority
plt.axhline(y=326, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
plt.text(daily_averages['date'].min(), 330, 'Majority (326 seats)', color='black', fontsize=11, va='bottom', ha='left', fontweight='bold')

plt.title('UK Parliamentary Seat Trends by Party', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Seats', fontsize=12)
plt.ylim(0, 550)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('seat_trends.png', dpi=300, bbox_inches='tight')
plt.show() 