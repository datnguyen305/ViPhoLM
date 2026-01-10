#!/bin/bash

echo "--- Đang khởi chạy Script ---"
export OG_DATA="../../../datasets/Wikilingual-dataset"
export OF_DATA="./data"

# Preprocess data
echo "--- Train Word2Vec model ---"
python3 preprocess_data/train_word2vec.py --path=word_embedding 

echo "--- Đang cho đúng định dạng data ---"
python3 preprocess_data/smaller_chunk.py

echo "--- Đang tạo pseudo-labels ---"
python3 make_extraction_labels.py

echo "--- Creating Vocab Size ---"
python3 preprocess_data/making_vocab.py

echo "--- Đang huấn luyện Abstractor ---"
python train_abstractor.py --path=pre-trained_models\abstractor --w2v=word_embedding\word2vec.128d.4k.bin

echo "--- Đang huấn luyện Extractor ---"
python train_extractor_ml.py --path=pre-trained_models\extractor --w2v=word_embedding\word2vec.128d.4k.bin

echo "--- Trainfull Model ---"
python train_full_rl.py --path=pre-trained_models\full_model --abs_dir=pre-trained_models\abstractor --ext_dir=pre-trained_models\extractor