from yacs.config import CfgNode
import yaml

def get_config(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f: # Thêm encoding="utf-8" tại đây
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return CfgNode(init_dict=cfg)
