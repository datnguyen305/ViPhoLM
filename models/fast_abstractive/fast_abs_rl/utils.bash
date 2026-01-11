#!/bin/bash
set -e   # dừng ngay nếu có lỗi

echo "--- W2V ---"

# Tạo thư mục trước
mkdir -p models/fast_abstractive/fast_abs_rl

# Download
gdown --id 1MPDzUzr7viJU4-MERmX900gZWZPGzs-u

echo "--- Unzip ---"
unzip word2vec_vi_words_100dims.zip
