import pandas as pd
import json
import os

# Paths to data files (adjust if needed)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/pj.stock/data-portfolio/data-portfolio/Seat_Calculator_Project'))
GE_RESULTS_PATH = os.path.join(BASE_DIR, 'GE_results.csv')
POLLS_PATH = os.path.join(os.path.dirname(__file__), '/Users/pj.stock/data-portfolio/data-portfolio/Polling_project/polls.csv')

party_columns = ["Con", "Lab", "LD", "RUK", "Green", "SNP", "PC", "All other candidates"]
rename_dict = {
    'Lab': 'Lab', 'Con': 'Con', 'Ref': 'RUK', 'LD': 'LD',
    'Greens': 'Green', 'SNP': 'SNP', 'PC': 'PC', 'Others': 'All other candidates'
}

# Load static GE results
results = pd.read_csv(GE_RESULTS_PATH)
for party in party_columns:
    results[party] = (results[party] / results["Valid votes"] * 100).round(2)
results_pct = results[["Valid votes", "Constituency name", "Country name"] + party_columns]

# 2024 GE national shares
GE_result = {
    "Lab": 33.7, "Con": 23.7, "LD": 12.2, "RUK": 14.3, "Green": 6.8, 
    "SNP": 2.5, "PC": 0.7, "All other candidates": 6.1
}

def seat_calc_for_poll(poll_row):
    poll_mapped = {}
    for party in party_columns:
        poll_value = poll_row.get(party)
        # Skip non-numeric, blank, or dash values
        try:
            if pd.notna(poll_value) and str(poll_value).strip() not in ('', '–', '-', 'Not Supplied'):
                poll_mapped[party] = float(poll_value)
        except ValueError:
            continue
    
    # Debug: print what was mapped
    print(f"Poll {poll_row.get('Pollster', 'Unknown')} on {poll_row.get('Date', 'Unknown')}:")
    print("  Poll mapped:", poll_mapped)
    
    swings = {party: poll_mapped[party] - GE_result.get(party, 0)
              for party in party_columns if party in poll_mapped}
    results_pct_adjusted = results_pct.copy()
    for party in party_columns:
        if party in swings:
            results_pct_adjusted[party] = results_pct_adjusted[party] + swings[party]
            results_pct_adjusted[party] = results_pct_adjusted[party].clip(lower=0)
    parties_with_data = [party for party in party_columns if party in poll_mapped]
    
    # Debug: print parties with data
    print("  Parties with data:", parties_with_data)
    
    # Check if we have any parties with data
    if len(parties_with_data) == 0:
        print("  Warning: No parties with polling data found - skipping this poll")
        return None  # Skip this poll
    
    results_pct_adjusted[parties_with_data] = (
        results_pct_adjusted[parties_with_data]
        .div(results_pct_adjusted[parties_with_data].sum(axis=1), axis=0) * 100
    ).round(6)
    results_pct_adjusted["Winner"] = results_pct_adjusted[parties_with_data].idxmax(axis=1)
    seat_counts = results_pct_adjusted["Winner"].value_counts().to_dict()
    output_data = {
        'pollster': str(poll_row.get('Pollster', 'Unknown')),
        'date': str(poll_row.get('Date', 'Unknown')),
        'seats': {
            'Reform UK': int(seat_counts.get('RUK', 0)),
            'Labour': int(seat_counts.get('Lab', 0)),
            'Conservative': int(seat_counts.get('Con', 0)),
            'Liberal Democrats': int(seat_counts.get('LD', 0)),
            'SNP': int(seat_counts.get('SNP', 0)),
            'Others': int(seat_counts.get('All other candidates', 0)),
            'Green': int(seat_counts.get('Green', 0)),
            'Plaid Cymru': int(seat_counts.get('PC', 0))
        }
    }
    return output_data

# Load and prepare polls
polls = pd.read_csv(POLLS_PATH)
polls = polls.rename(columns=rename_dict)

results_list = []
for idx, row in polls.iterrows():
    result = seat_calc_for_poll(row)
    if result:
        results_list.append(result)

# Convert to DataFrame for easy analysis
results_df = pd.DataFrame(results_list)
results_df.to_csv('all_seat_results.csv', index=False)
results_df.to_json('all_seat_results.json', orient='records', indent=2)

print(f"Processed {len(results_df)} polls. Results saved to all_seat_results.csv and all_seat_results.json.") 