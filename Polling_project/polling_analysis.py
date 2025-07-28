import pandas as pd
import numpy as np
from plotnine import *
import json

# Read the CSV file
df = pd.read_csv("polls.csv")

# Convert date column to datetime
# (add .str.strip() for safety)
df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%d/%m/%Y', errors='coerce')
df['Date_Numeric'] = (df['Date'] - df['Date'].min()).dt.days

# Group by date and calculate mean for each party
party_columns = ['Lab', 'Con', 'Ref', 'LD', 'Greens', 'SNP', 'PC', 'Others']

# Replace empty strings and whitespace with NaN
for col in party_columns:
    df[col] = pd.to_numeric(df[col].replace(r'^\s*$', np.nan, regex=True), errors='coerce')

# Calculate daily averages, excluding NaN values
daily_averages = df.groupby('Date')[party_columns].mean().reset_index()

df_long = daily_averages.melt(id_vars=['Date'], value_vars=party_columns, var_name='Party', value_name='Support')
df_long = df_long.dropna(subset=['Support'])

# Define party colors for plotnine
ggplot_colors = {
    'Lab': '#E4003B',
    'Con': '#0087DC',
    'Ref': '#12B6CF',
    'LD': '#FAA61A',
    'Greens': '#6AB023',
    'SNP': '#000000',
    'PC': '#3F8428',
    'Others': '#999999'
}

# Create the plot using plotnine (ggplot style)
p = (
    ggplot(df_long, aes('Date', 'Support', color='Party'))
    + geom_point(alpha=0.5)
    + geom_smooth(method='loess', span=0.2, se=False)
    + scale_color_manual(values=ggplot_colors)
    + scale_y_continuous(limits=[0, 40])
    + labs(
        title='Westminster Voting Intentions (GB)',
        x='Date',
        y='Support (%)'
    )
    + geom_vline(xintercept=pd.to_datetime('01/05/2025', format='%d/%m/%Y'), linetype='dashed', color='black', alpha=0.5)
    + annotate('text', x=pd.to_datetime('01/05/2025', format='%d/%m/%Y'), y=33, label='Local Elections - England', angle=45, va='bottom', ha='left', size=8)
    + scale_x_datetime(date_breaks='1 month', date_labels='%b %Y')
    + theme_bw()
    + theme(
        axis_text_x=element_text(rotation=45, ha='right'),
        figure_size=(14, 8),
        legend_title=element_text(size=12),
        legend_position='bottom'
    )
)

# Save the plot
p.save('polling_trends.png', dpi=300, bbox_inches='tight')

# Calculate statistics
averages = df[party_columns].mean().round(1).sort_values(ascending=False)
latest_date = daily_averages['Date'].max()
latest_polls = daily_averages[daily_averages['Date'] == latest_date]

most_recent_poll = df.sort_values('Date', ascending=False).iloc[0]
most_recent_date = most_recent_poll['Date']

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

with open('polling_data.json', 'w') as f:
    json.dump(polling_data, f, indent=4)

print("\nAverage support by party:")
print(averages)

print("\nLatest polling averages:")
print(f"\nAs of {latest_date.strftime('%d %B %Y')}:" )
latest_values = latest_polls[party_columns].iloc[0]
sorted_latest = latest_values.sort_values(ascending=False)
for party, value in sorted_latest.items():
    if pd.notna(value):
        print(f"{party}: {value:.1f}%") 