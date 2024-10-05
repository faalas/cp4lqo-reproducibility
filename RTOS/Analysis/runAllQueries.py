import os
import sys
from tqdm import tqdm  # 导入 tqdm 进度条库

from PGUtils import pgrunner
from sqlSample import sqlInfo
from itertools import count
from DQN import DQN, ENV
from TreeLSTM import SPINN
from JOBParser import DB
from ImportantConfig import Config
import torch

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() and config.usegpu == 1 else "cpu")

with open(config.schemaFile, "r") as f:
    createSchema = "".join(f.readlines())

db_info = DB(createSchema)

featureSize = 128

policy_net = SPINN(n_classes=1, size=featureSize, n_words=100, mask_size=40 * 41, device=device).to(device)
target_net = SPINN(n_classes=1, size=featureSize, n_words=100, mask_size=40 * 41, device=device).to(device)
policy_net.load_state_dict(torch.load("LatencyTuning.pth"))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

DQN = DQN(policy_net, target_net, db_info, pgrunner, device)


def load_sql_files_to_map(folder_path):
    sql_map = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.sql'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                sql_content = file.read().strip()

            sql_map[filename] = sql_content

    return sql_map


if __name__ == '__main__':
    log_file_path = '/RTOS/Analysis/execution_log_all.txt'

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        sys.stdout = log_file  

        hashmap = load_sql_files_to_map("/users/hanwen/workspace/RTOS/onelineQueries")

        for id, query in tqdm(hashmap.items(), desc="Processing SQL Queries", total=len(hashmap)):
            print(f"ID (Filename): {id}")
            print(f"Query: {query}")
            sqlSample = sqlInfo(pgrunner, query, "input")
            env = ENV(sqlSample, db_info, pgrunner, device, run_mode=True)
            print("-----------------------------")
            for t in count():
                action_list, chosen_cost, chosen_action, all_action = DQN.hanwen_select_action(env, need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left, right)
                # print("left: ", left, "right: ", right)
                print("chosen_action: ", chosen_action, ", chosen_cost: ", chosen_cost)
                reward, done = env.reward_new()
                if done:
                    for row in reward:
                        print(row)
                    break
            print("-----------------------------")

    sys.stdout = sys.__stdout__

    print(f"Logs have been saved to {log_file_path}")
