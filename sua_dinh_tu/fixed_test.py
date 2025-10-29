import json
file_names = ['test']
max_files = 10  # Số file đầu tiên bạn muốn in ra
file_count = 0
total = 0
output_dir = 'datasets/UIT/fixed_test.json'

for file_name in file_names:
    file_directory = f'datasets/UIT/testing {file_name}.json'
    with open(file_directory, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for id, content in data.items():
        total += 1
        print(f"File: {file_name}, ID: {id}")
        for key, value in content.items():
            if key == 'source':
                for i, sentence in value.items():
                    value[i] = list(set(sentence))
                        
# Write the DIC dictionary to a JSON file
with open(output_dir, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f'Data successfully exported to {output_dir}')