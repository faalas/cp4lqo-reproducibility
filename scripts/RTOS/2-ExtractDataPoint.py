import os
import re
import csv
import shutil

def process_log_file(input_file, output_file):
    # Regular expressions to match required data
    action_pattern = r"Chosen action: \['(.*?)', '(.*?)'\]"  # Match Chosen action
    action_cost_pattern = r"Chosen cost: (-?[\d.]+)"  # Match Chosen cost, which can be negative
    cost_pattern = r"Cost: (-?[\d.]+)"  # Match Cost, which can be negative
    runtime_pattern = r"Actual time: (-?[\d.]+)"  # Match Actual time, which can be negative

    # Read file content
    with open(input_file, 'r') as file:
        file_content = file.read()

    # Separate the part before "Original Log"
    parts = file_content.split("Original Log:")
    log_part = parts[0].strip() if len(parts) > 1 else file_content

    # Extract the required data
    actions = re.findall(action_pattern, log_part)  # Extract Chosen action
    action_costs = re.findall(action_cost_pattern, log_part)  # Extract Chosen cost
    costs = re.findall(cost_pattern, log_part)  # Extract Cost
    runtimes = re.findall(runtime_pattern, log_part)  # Extract Actual time

    # Check for consistency in lengths
    if not (len(actions) == len(action_costs) == len(costs) == len(runtimes)):
        print("Error: Mismatch in lengths of extracted data.")
        print(input_file)
        exit(1)

    # Write data to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write CSV header
        csvwriter.writerow(["action", "action_cost", "cost", "runtime"])

        # Write each row of data
        for i in range(len(actions)):
            action = f"{actions[i][0]} -> {actions[i][1]}"
            csvwriter.writerow([action, action_costs[i], costs[i], runtimes[i]])

    print(f"Data successfully written to {output_file}")

# Define input and output folders
input_folder = "./Analysis/1-extraction"
output_folder = "./Analysis/2-datapoints"

# Delete and recreate the output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Process files
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")

        # Process log file and generate CSV
        process_log_file(input_file_path, output_file_path)
