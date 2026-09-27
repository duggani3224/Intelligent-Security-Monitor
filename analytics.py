import csv
import os

# CSV file created in Step 11
csv_file = "events.csv"

if not os.path.exists(csv_file):
    print("ERROR: events.csv not found")
    print("Run save_events.py first.")
    exit()

# Lists for storing data
entries = []
exits = []
dwell_times = []

# Read events.csv
with open(csv_file, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        event = row["event"]

        # Count entries
        if event == "ENTRY":
            entries.append(row)

        # Count exits and collect dwell times
        elif event == "EXIT":
            exits.append(row)

            if row["duration_s"]:
                dwell_times.append(float(row["duration_s"]))


# Calculate statistics
total_entries = len(entries)
total_exits = len(exits)

if dwell_times:
    average_dwell_time = sum(dwell_times) / len(dwell_times)
else:
    average_dwell_time = 0


# Find maximum ROI occupancy
current_inside = 0
maximum_roi_count = 0

for row in entries:
    current_inside += 1

    if current_inside > maximum_roi_count:
        maximum_roi_count = current_inside

for row in exits:
    current_inside -= 1


# Display analytics
print()
print("===================================")
print("   INTELLIGENT SECURITY MONITOR")
print("          ANALYTICS")
print("===================================")

print(f"Total Entries       : {total_entries}")
print(f"Total Exits         : {total_exits}")
print(f"Maximum ROI Count   : {maximum_roi_count}")
print(f"Average Dwell Time  : {average_dwell_time:.2f} seconds")

print("===================================")
print("Analytics calculation completed.")