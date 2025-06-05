import pandas as pd
import json

# Load data
results = pd.read_csv("GE_results.csv")
party_columns = ["Con", "Lab", "LD", "RUK", "Green", "SNP", "PC", "All other candidates"]
polls = pd.read_csv("/Users/pj.stock/data-portfolio/data-portfolio/Polling_project/polls.csv")



rename_dict = {
    'Lab': 'Lab', 'Con': 'Con', 'Ref': 'RUK', 'LD': 'LD',
    'Greens': 'Green', 'SNP': 'SNP', 'PC': 'PC', 'Others': 'All other candidates'
}


# Rename poll columns
rename_dict = {
    'Lab': 'Lab', 'Con': 'Con', 'Ref': 'RUK', 'LD': 'LD',
    'Greens': 'Green', 'SNP': 'SNP', 'PC': 'PC', 'Others': 'All other candidates'
}
polls = polls.rename(columns=rename_dict)

# Select the latest poll
latest_poll = polls.iloc[0]


# Map poll to party_columns
poll_mapped = {
    "Con": float(latest_poll["Con"]),
    "Lab": float(latest_poll["Lab"]),
    "LD": float(latest_poll["LD"]),
    "RUK": float(latest_poll["RUK"]),
    "Green": float(latest_poll["Green"]),
    "SNP": float(latest_poll["SNP"]),
    "PC": float(latest_poll.get("PC", 0)),
    "All other candidates": float(latest_poll["All other candidates"])
}

# 2024 GE national shares
GE_result = {
    "Lab": 33.7, "Con": 23.7, "LD": 12.2, "RUK": 14.3, "Green": 6.8, 
    "SNP": 2.5, "PC": 0.7, "All other candidates": 6.1
}

# Calculate vote share percentages
for party in party_columns:
    results[party] = (results[party] / results["Valid votes"] * 100).round(2)

results_pct = results[["Valid votes", "Constituency name", "Country name", "Con", "Lab", "LD", "RUK", "Green", "SNP", "PC", "All other candidates"]]

# Calculate swings
swings = {party: poll_mapped[party] - GE_result.get(party, 0) for party in party_columns}

# Apply swings
results_pct_adjusted = results_pct.copy()
for party in party_columns:
    if party not in ["SNP", "PC"]:
        results_pct_adjusted[party] = results_pct_adjusted[party] + swings[party]
        results_pct_adjusted[party] = results_pct_adjusted[party].clip(lower=0)

# Normalize percentages to sum to 100% and round to avoid precision issues
results_pct_adjusted[party_columns] = (results_pct_adjusted[party_columns]
                                      .div(results_pct_adjusted[party_columns].sum(axis=1), axis=0) * 100).round(6)


# Determine winners
results_pct_adjusted["Winner"] = results_pct_adjusted[party_columns].idxmax(axis=1)

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
    'swings': {
        'Labour': {
            'ge_share': GE_result['Lab'],
            'latest_share': float(latest_poll['Lab']),
            'swing': float(swings['Lab'])
        },
        'Conservative': {
            'ge_share': GE_result['Con'],
            'latest_share': float(latest_poll['Con']),
            'swing': float(swings['Con'])
        },
        'Reform UK': {
            'ge_share': GE_result['RUK'],
            'latest_share': float(latest_poll['RUK']),
            'swing': float(swings['RUK'])
        },
        'Liberal Democrats': {
            'ge_share': GE_result['LD'],
            'latest_share': float(latest_poll['LD']),
            'swing': float(swings['LD'])
        },
        'Green': {
            'ge_share': GE_result['Green'],
            'latest_share': float(latest_poll['Green']),
            'swing': float(swings['Green'])
        },
        'Others': {
            'ge_share': GE_result['All other candidates'],
            'latest_share': float(latest_poll['All other candidates']),
            'swing': float(swings['All other candidates'])
        }
    }
}

# Verify JSON matches printed seat counts
print("\nJSON Output:", json.dumps(output_data, indent=4))

# Save to JSON
with open('seat_results.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("Results saved to seat_results.json")