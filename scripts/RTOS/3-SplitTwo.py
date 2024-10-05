import os
import re
import csv
import shutil

def process_log_file(input_file, output_dir):
    # Regular expressions to match required data, including negative numbers
    action_cost_pattern = r"Chosen cost: (-?[\d.]+)"  # Match Chosen cost, which can be negative
    cost_pattern = r"Cost: (-?[\d.]+)"                # Match Cost, which can be negative
    runtime_pattern = r"Actual time: (-?[\d.]+)"      # Match Actual time, which can be negative

    # Read file content
    with open(input_file, 'r') as file:
        file_content = file.read()

    # Separate the part before "Original Log"
    parts = file_content.split("Original Log:")
    log_part = parts[0].strip() if len(parts) > 1 else file_content

    # Extract the required data
    action_costs = re.findall(action_cost_pattern, log_part)
    costs = re.findall(cost_pattern, log_part)
    runtimes = re.findall(runtime_pattern, log_part)

    # Check for consistency in lengths
    if not (len(action_costs) == len(costs) == len(runtimes)):
        print(f"Error: Mismatch in lengths of extracted data for file: {input_file}")
        return

    # Write to the first CSV file: action_cost and runtime
    action_cost_output_file = os.path.join(output_dir, f"action_cost_{os.path.splitext(os.path.basename(input_file))[0]}.csv")
    with open(action_cost_output_file, 'w', newline='') as csvfile1:
        csvwriter1 = csv.writer(csvfile1)
        # Write CSV header
        csvwriter1.writerow(["action_cost", "runtime"])

        # Write each row of data
        for i in range(len(action_costs)):
            csvwriter1.writerow([action_costs[i], runtimes[i]])

    print(f"Action cost and runtime data successfully written to {action_cost_output_file}")

    # Write to the second CSV file: cost and runtime
    cost_output_file = os.path.join(output_dir, f"cost_{os.path.splitext(os.path.basename(input_file))[0]}.csv")
    with open(cost_output_file, 'w', newline='') as csvfile2:
        csvwriter2 = csv.writer(csvfile2)
        # Write CSV header
        csvwriter2.writerow(["cost", "runtime"])

        # Write each row of data
        for i in range(len(costs)):
            csvwriter2.writerow([costs[i], runtimes[i]])

    print(f"Cost and runtime data successfully written to {cost_output_file}")

# Define input and output folders
input_folder = "./Analysis/1-extraction"
output_folder = "./Analysis/3-Separate"

# Delete and recreate the output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

# Process files
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):  # Process .txt files
        input_file_path = os.path.join(input_folder, filename)

        # Process log file and generate CSV
        process_log_file(input_file_path, output_folder)
