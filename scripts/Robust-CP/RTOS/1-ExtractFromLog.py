import re
import os
import shutil

TARGET_JOINS = ["Nested Loop", "Hash Join", "Parallel Hash Join", "Merge Join"]

def process_sql_log(file_content, output_dir):
    # Create output directory (if it doesn't exist)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    # Extract each SQL's ID
    sql_id_pattern = r"ID \(Filename\):\s*(\S+)"
    sql_blocks = re.split(sql_id_pattern, file_content)
    # print("SQL Blocks: ", sql_blocks)

    # Extract chosen_action and chosen_cost
    action_pattern = r"chosen_action:\s*\((.*?)\)\s*,\s*chosen_cost:\s*tensor\(\[\[(.*?)\]\],.*?\)"

    # Extract cost and actual time for join operations
    join_pattern = r"->\s*(.*?)\s*\(cost=(.*?)\.\.(.*?) rows.*?width.*?actual time=(.*?)\.\.(.*?) rows=.*?loops=(\d+)"

    # Process each SQL block
    for i in range(1, len(sql_blocks), 2):
        sql_id = sql_blocks[i].strip()  # Get the SQL ID
        # print(sql_id)
        sql_content = sql_blocks[i + 1]  # Get the content of the SQL
        # print(sql_content)
        actions = re.findall(action_pattern, sql_content)  # Find chosen_action and chosen_cost

        all_joins = re.findall(join_pattern, sql_content)  # Find all join operations
        joins = [join for join in all_joins if join[0] in TARGET_JOINS]
        joins = joins[::-1]

        output_lines = []  # Store calculated results
        raw_log = []  # Store raw logs

        # Match each chosen_action with corresponding join and output results
        try:
            # Ensure chosen_action and join match in count
            assert (len(actions) == len(joins)), f"Mismatch between actions and joins for SQL ID: {sql_id}"
        except AssertionError as e:
            print(f"\n\nAssertion failed for SQL ID: {sql_id}. Error: {e}")
            print(len(actions))
            print(len(joins))
            for i in range(min(len(actions), len(joins))):
                print(actions[i], joins[i])
            print(all_joins)

            return  # Terminate processing
        for j in range(min(len(actions), len(joins))):
            action = actions[j]
            join = joins[j]
            # Get values after cost and actual time
            output_lines.append(
                f"Chosen action: [{action[0]}], Chosen cost: {action[-1]}, Join type: {join[0]}, Cost: {join[2]}, Actual time: {join[4]}")

        # Construct output file content
        output_file_content = "\n".join(output_lines) + "\n\nOriginal Log:\n" + sql_content

        # Output to a separate file
        output_file = os.path.join(output_dir, f"{sql_id}.txt")
        with open(output_file, 'w') as f:
            f.write(output_file_content)

        print(f"Processed {sql_id}, output saved to {output_file}")

# Read the file
with open("LIGHT_execution_log_all.txt", "r") as f:
# with open("JOB_execution_log_all.txt", "r") as f:
    file_content = f.read()

# Define output directory
output_dir = "./[LIGHT]1-extraction"
# output_dir = "./[JOB]1-extraction"

# Process file content and save to the folder
process_sql_log(file_content, output_dir)
