import os

input_folder = '/RTOS/Queries'
output_folder = '/RTOS/onelineQueries'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.sql'):
        input_file_path = os.path.join(input_folder, filename)
        with open(input_file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()

        oneline_sql = ' '.join(sql_content.split())

        output_file_path = os.path.join(output_folder, filename)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(oneline_sql)

        print(f'Processed {filename} into one line and saved to {output_file_path}')
