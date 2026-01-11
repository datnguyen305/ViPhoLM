#!/bin/bash

echo "--- Đang khởi chạy Script ---"
export OG_DATA="../../../datasets/Vietnews-dataset"
export OF_DATA="./data"
export ROUGE="/home/jovyan/pyrouge/tools/ROUGE-1.5.5"
export WNDB=/home/jovyan/pyrouge/tools/ROUGE-1.5.5/data
# Preprocess data
echo "--- Đang cho đúng định dạng data ---"
python3 preprocess_data/smaller_chunk.py

echo "--- Đang tạo pseudo-labels ---"
python3 make_extraction_labels.py

echo "--- Train Word2Vec model ---"
python3 train_word2vec.py --path=word_embedding_Vietnews

echo "--- Creating Vocab Size ---"
python3 preprocess_data/making_vocab.py

echo "--- Đang huấn luyện Abstractor ---"
python train_abstractor.py --path=pre-trained_models_Vietnews/abstractor --w2v=word_embedding_Vietnews/word2vec.128d.10k.bin

echo "--- Đang huấn luyện Extractor ---"
python train_extractor_ml.py --path=pre-trained_models_Vietnews/extractor --w2v=word_embedding_Vietnews/word2vec.128d.10k.bin

echo "--- Trainfull Model ---"
python train_full_rl.py --path=pre-trained_models_Vietnews/full_model --abs_dir=pre-trained_models_Vietnews/abstractor --ext_dir=pre-trained_models_Vietnews/extractor

echo "--- Decode test ---"
python decode_full_model.py --path=result/Vietnews --model_dir=pre-trained_models_Vietnews/full_model --beam=1 --test

echo "-- Make eval references ---"
python make_eval_references.py

echo "-- Download Rouge --"
cd /home/jovyan/
git clone https://github.com/andersjo/pyrouge.git
cd /home/jovyan/pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions
perl buildExeptionDB.pl . exc ../WordNet-2.0.exc.db

echo "-- Evaluate full model ---"
cd /home/jovyan/TestingViPho/models/fast_abstractive/fast_abs_rl
python eval_full_model.py --rouge --decode_dir=result
