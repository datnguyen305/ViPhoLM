import os
import json
import pickle
import tarfile
import re
from collections import Counter
from os.path import join, dirname
import unicodedata

# 1. Kiểm tra biến môi trường
OG_DATA_DIR = os.environ.get('OG_DATA')
OF_DATA_DIR = os.environ.get('OF_DATA')

if not OG_DATA_DIR or not OF_DATA_DIR:
    print('LỖI: Thiếu biến môi trường OG_DATA hoặc OF_DATA. Hãy kiểm tra lại main.bash')
    exit(1)

def simple_tokenize(sentence):
    sentence = sentence.lower()
    sentence = unicodedata.normalize("NFD", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    sentence = re.sub(r"%", " % ", sentence)
    sentence = re.sub(r"<nl>", " <nl> ", sentence) # new line mark

    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()

    return tokens

def process_and_build():
    word_cnt = Counter()
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        input_file = join(OG_DATA_DIR, f"{split}.json")
        output_dir = join(OG_DATA_DIR, split) # Lưu các file nhỏ tại OG_DATA_DIR
        
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Đang xử lý tập {split} (Tách âm tiết đơn)...")
        
        if not os.path.exists(input_file):
            print(f"Bỏ qua {split} vì không tìm thấy file {input_file}")
            continue

        with open(input_file, 'r', encoding='utf-8') as f:
            big_data = json.load(f)
            
        for i, (key, value) in enumerate(big_data.items()):
            raw_source = value['source'].get('0', []) if isinstance(value['source'], dict) else value['source']
            raw_target = value['target']

            source_sents = [" ".join(simple_tokenize(s)) for s in raw_source]
            target_text = [" ".join(simple_tokenize(raw_target))]
            
            for sent in source_sents + target_text:
                word_cnt.update(sent.split())
            
            single_data = {
                "article": source_sents, 
                "abstract": target_text
            }
            with open(join(output_dir, f'{i}.json'), 'w', encoding='utf-8') as f_out:
                json.dump(single_data, f_out, ensure_ascii=False, indent=4)

    # 2. Lưu vocab_cnt.pkl (Đảm bảo tạo thư mục OF_DATA_DIR trước)
    os.makedirs(OF_DATA_DIR, exist_ok=True)
    vocab_path = join(OF_DATA_DIR, 'vocab_cnt.pkl')
    with open(vocab_path, 'wb') as f_vocab:
        pickle.dump(word_cnt, f_vocab)
    print(f"--- Đã tạo xong vocab_cnt.pkl tại: {vocab_path}. Tổng từ: {len(word_cnt)} ---")

    # 3. Đóng gói thành Tarballs (.tar)
    for split in splits:
        # Lấy dữ liệu từ OG_DATA_DIR (nơi vừa lưu file JSON nhỏ)
        source_folder = join(OG_DATA_DIR, split)
        # Lưu file .tar vào OF_DATA_DIR
        tar_path = join(OF_DATA_DIR, f"{split}.tar")
        
        if os.path.exists(source_folder):
            print(f"Đang đóng gói {tar_path}...")
            with tarfile.open(tar_path, "w") as tar:
                # arcname=split giúp khi giải nén ra sẽ là một thư mục mang tên split
                tar.add(source_folder, arcname=split)
            
    print("--- HOÀN TẤT QUY TRÌNH TIỀN XỬ LÝ ---")

if __name__ == '__main__':
    process_and_build()