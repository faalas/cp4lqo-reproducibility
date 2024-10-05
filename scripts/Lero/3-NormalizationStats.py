import os
import numpy as np
import csv

def scan_folder_and_calculate(input_folder):
    cost_runtime_ratios = []
    cost_log = []

    # Iterate over all CSV files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)

            # Open and read the CSV file
            with open(file_path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)

                # Iterate over each row in the CSV file
                for row in csvreader:
                    try:
                        cost = float(row['Cost'])
                        runtime = float(row['Actual Total Time'])
                        cost_log.append(np.log(cost))

                        # Ensure runtime is not zero
                        if runtime > 0:
                            # Calculate the cost/runtime ratio
                            ratio = cost / runtime
                            cost_runtime_ratios.append(ratio)

                            log_ratio = np.log(cost) - np.log(runtime)
                            print(filename, cost, runtime, ratio, log_ratio)

                        else:
                            print(f"Warning: Runtime is zero in file {filename}")
                    except ValueError:
                        # If parsing fails, continue with other rows
                        print(f"Warning: Could not parse values in {filename}")
                    break

    # Calculate the average cost/runtime ratio
    if cost_runtime_ratios:
        print(cost_runtime_ratios)
        average_ratio = sum(cost_runtime_ratios) / len(cost_runtime_ratios)
        print(f"Average Cost/Runtime Ratio: {average_ratio}")

        print("log_cost: ", cost_log)
        average_log = sum(cost_log) / len(cost_log)
        print(average_log)
    else:
        print("No valid cost/runtime ratios to calculate.")

# Define input folder
input_folder = "./TPCH/"  # Replace with the actual folder path
input_folder = "./IMDB/"  # Replace with the actual folder path

# Scan folder and calculate each cost/runtime ratio and its average
scan_folder_and_calculate(input_folder)
