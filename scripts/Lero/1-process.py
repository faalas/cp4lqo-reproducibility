## Only need to change the top filename and output_dir

import json
import csv
import os

filename = ""
output_dir = "./IMDB/Guarantee"

filename = ""
output_dir = "./TPCH/Guarantee"

def clear_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    else:
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

excluded_node_types = {"Index Scan", "Seq Scan", "Materialize", "Hash", "Index Only Scan"}

def extract_plan_info(plan, results):
    node_type = plan.get('Node Type', 'Unknown')
    
    if node_type in excluded_node_types:
        return
    
    actual_total_time = plan.get('Actual Total Time', 'N/A')
    total_cost = plan.get('Total Cost', 'N/A')

    results.append({
        'Node Type': node_type,
        'Actual Total Time': actual_total_time,
        'Total Cost': total_cost
    })

    if 'Plans' in plan:
        for sub_plan in plan['Plans']:
            extract_plan_info(sub_plan, results)

def process_explain_analyze_output(file_name, output_dir):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        sql_name, json_content = line.split('#####')
        base_name = sql_name.split('.sql')[0] 
        explain_data = json.loads(json_content)

        results = []
        extract_plan_info(explain_data[0]['Plan'], results)

        csv_file_path = os.path.join(output_dir, f"{base_name.strip()}.csv")

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Operation", "Actual Total Time", "Cost"])

            for plan in results:
                csv_writer.writerow([plan['Node Type'], plan['Actual Total Time'], plan['Total Cost']])

        print(f"Results written to {csv_file_path}")


clear_output_dir(output_dir)
process_explain_analyze_output(filename, output_dir)
