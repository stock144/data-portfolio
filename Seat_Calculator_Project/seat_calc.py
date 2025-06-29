import pandas as pd
import json

# Load data
results = pd.read_csv("GE_results.csv")
party_columns = ["Con", "Lab", "LD", "RUK", "Green", "SNP", "PC", "All other candidates"]
polls = pd.read_csv("/Users/pj.stock/data-portfolio/data-portfolio/Polling_project/polls.csv")

# Rename poll columns
rename_dict = {
    'Lab': 'Lab', 'Con': 'Con', 'Ref': 'RUK', 'LD': 'LD',
    'Greens': 'Green', 'SNP': 'SNP', 'PC': 'PC', 'Others': 'All other candidates'
}
polls = polls.rename(columns=rename_dict)

# Select the latest poll
latest_poll = polls.iloc[0]

# Map poll to party_columns, handling NaN values
poll_mapped = {}
for party in party_columns:
    poll_value = latest_poll.get(party)
    if pd.notna(poll_value):
        poll_mapped[party] = float(poll_value)

# Debug: print what was mapped
print(f"Poll {latest_poll.get('Pollster', 'Unknown')} on {latest_poll.get('Date', 'Unknown')}:")
print("Poll mapped:", poll_mapped)

# 2024 GE national shares
GE_result = {
    "Lab": 33.7, "Con": 23.7, "LD": 12.2, "RUK": 14.3, "Green": 6.8, 
    "SNP": 2.5, "PC": 0.7, "All other candidates": 6.1
}

# Calculate vote share percentages
for party in party_columns:
    results[party] = (results[party] / results["Valid votes"] * 100).round(2)

results_pct = results[["Valid votes", "Constituency name", "Country name", "Con", "Lab", "LD", "RUK", "Green", "SNP", "PC", "All other candidates"]]

# Calculate swings only for parties with polling data
swings = {}
for party in party_columns:
    if party in poll_mapped:
        swings[party] = poll_mapped[party] - GE_result.get(party, 0)

# Apply swings only to parties with polling data
results_pct_adjusted = results_pct.copy()
for party in party_columns:
    if party in swings:
        results_pct_adjusted[party] = results_pct_adjusted[party] + swings[party]
        results_pct_adjusted[party] = results_pct_adjusted[party].clip(lower=0)

# Normalize percentages to sum to 100% and round to avoid precision issues
# Only include parties with polling data in the normalization
parties_with_data = [party for party in party_columns if party in poll_mapped]

# Debug: print parties with data
print("Parties with data:", parties_with_data)

# Check if we have any parties with data
if len(parties_with_data) == 0:
    print("Warning: No parties with polling data found")
    print("Available poll data:", poll_mapped)
    print("Available columns in latest_poll:", latest_poll.index.tolist())
    exit(1)  # Exit with error

results_pct_adjusted[parties_with_data] = (results_pct_adjusted[parties_with_data]
                                      .div(results_pct_adjusted[parties_with_data].sum(axis=1), axis=0) * 100).round(6)

# Determine winners only among parties with polling data
results_pct_adjusted["Winner"] = results_pct_adjusted[parties_with_data].idxmax(axis=1)

# Aggregate seat counts
seat_counts = results_pct_adjusted["Winner"].value_counts().to_dict()

# Print results
print("Swings (%):", pd.Series(swings).round(2))
print("\nPredicted Seat Counts:", seat_counts)

# Prepare JSON output
output_data = {
    'pollster': str(latest_poll.get('Pollster', 'Unknown')),
    'date': str(latest_poll.get('Date', 'Unknown')),
    'seats': {
        'Reform UK': int(seat_counts.get('RUK', 0)),
        'Labour': int(seat_counts.get('Lab', 0)),
        'Conservative': int(seat_counts.get('Con', 0)),
        'Liberal Democrats': int(seat_counts.get('LD', 0)),
        'SNP': int(seat_counts.get('SNP', 0)),
        'Others': int(seat_counts.get('All other candidates', 0)),
        'Green': int(seat_counts.get('Green', 0)),
        'Plaid Cymru': int(seat_counts.get('PC', 0))
    },
    'swings': {}
}

# Add swing data only for parties with polling data
for party, swing in swings.items():
    party_name = {
        'Lab': 'Labour',
        'Con': 'Conservative',
        'RUK': 'Reform UK',
        'LD': 'Liberal Democrats',
        'Green': 'Green',
        'SNP': 'SNP',
        'PC': 'Plaid Cymru',
        'All other candidates': 'Others'
    }[party]
    
    output_data['swings'][party_name] = {
        'ge_share': GE_result[party],
        'latest_share': float(poll_mapped[party]),
        'swing': float(swing)
    }

# Save to JSON
with open('seat_results.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("Results saved to seat_results.json")