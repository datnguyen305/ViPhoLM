import json
import os

files = ['dev', 'train', 'test']
try: 
    OG_DATA_DIR = os.environ['OG_DATA']
except KeyError:
    print('please use environment variable to specify data directories')

try:
    OF_DATA_DIR = os.environ['OF_DATA']
except KeyError:
    print('please use environment variable to specify data directories')

for file in files: 
    input_file = rf'{OG_DATA_DIR}/{file}.json'
    # Thư mục DATA bạn đã thiết lập
    output_dir = rf'{OF_DATA_DIR}/{file}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r', encoding='utf-8') as f:
        big_data = json.load(f)

    # Lặp qua từng bài viết và lưu thành file riêng lẻ
    for i, (key, value) in enumerate(big_data.items()):
        # Chuyển đổi cấu trúc source từ dict {"0": [...]} thành list [...]
        article_sents = value['source']['0']
        abstract_text = value['target']
        
        # Tạo object mới đúng cấu trúc script mong đợi
        single_data = {
            "source": article_sents,  # Script của bạn đang dùng key 'source'
            "target": [abstract_text] # Đưa target vào list vì script tokenize theo list
        }
        
        with open(os.path.join(output_dir, f'{i}.json'), 'w', encoding='utf-8') as f:
            json.dump(single_data, f, indent=4, ensure_ascii=False)

    print(f"Đã tạo xong {len(big_data)} file JSON tại {output_dir}")