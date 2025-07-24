import json
from collections import Counter

# Paths to your JSON files
files = [
    "England_Polling/seat_results.json",
    "Welsh_Polling/seat_results.json",
    "Scottish_Polling/seat_results.json"
]

total_seats = Counter()

for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        # If data is a list, use only the first result
        if isinstance(data, list):
            data = data[0]
        # Extract the 'seats' dictionary
        seats = data.get("seats", {})
        # Only sum numeric values
        seats = {k: v for k, v in seats.items() if isinstance(v, (int, float))}
        total_seats.update(seats)

total_seats = dict(total_seats)

# Print the combined results
print("Combined seat totals (first result from each file):")
for party, seats in total_seats.items():
    print(f"{party}: {seats}")

# Save to a new JSON file
with open("UK_seat_results.json", "w") as f:
    json.dump(total_seats, f, indent=2) 