import yaml
import os

# Tự động load file YAML làm config mặc định
yaml_path = os.path.join(os.path.dirname(__file__),'HAT_VietNews.yaml')
with open(yaml_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
