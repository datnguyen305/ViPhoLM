import pandas as pd

# Ví dụ đọc dữ liệu từ file CSV
train_data = pd.read_json('train.csv', encoding='utf-8')
dev_data = pd.read_csv('dev.csv', encoding='utf-8')
test_data = pd.read_csv('test.csv', encoding='utf-8')
