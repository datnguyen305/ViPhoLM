import os
import json
import torch
import argparse
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from builders.vocab_builder import build_vocab
from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from data_utils import collate_fn_oov
from utils.instance import Instance
import evaluation

class BottomUpInference:
    def __init__(self, cs_config, pg_config, threshold=0.3, soft_coeff=0.15, device='cuda'):
        self.threshold = threshold
        self.soft_coeff = soft_coeff # Hệ số cho Soft Masking
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        with open(cs_config, 'r', encoding='utf-8') as f:
            self.cs_config = self._load_config(yaml.safe_load(f))
        with open(pg_config, 'r', encoding='utf-8') as f:
            self.pg_config = self._load_config(yaml.safe_load(f))

        print(f"Building vocabulary...")
        self.vocab = build_vocab(self.pg_config.vocab)

        cs_path = "checkpoints/content_selector/ContentSelector_Wikilingual/best_model.pth"
        pg_path = "checkpoints/pointer_generator_Wikilingual/best_model.pth"

        self.content_selector = self._load_model_weights(self.cs_config, cs_path, "Content Selector")
        self.pointer_generator = self._load_model_weights(self.pg_config, pg_path, "Pointer Generator")

        self.test_dataset = build_dataset(self.pg_config.dataset.test, self.vocab)
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_oov
        )

    def _load_config(self, data):
        if isinstance(data, dict):
            inst = Instance()
            for k, v in data.items(): setattr(inst, k, self._load_config(v))
            return inst
        return [self._load_config(i) for i in data] if isinstance(data, list) else data

    def _load_model_weights(self, config, path, name):
        config.model.device = self.device
        model = build_model(config.model, self.vocab)
        if os.path.exists(path):
            print(f"-> Loading {name} from {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"   SUCCESS: {name} weights loaded.")
        return model.to(self.device).eval()

    def apply_bottom_up_mask(self, model, input_ids, input_lengths, ext_idx, extra_zeros):
        with torch.no_grad():
            cs_input = input_ids.clone()
            cs_input[cs_input >= self.vocab.vocab_size] = self.vocab.unk_idx

            # 1. Lấy xác suất từ Content Selector
            selection_probs = self.content_selector(cs_input, input_lengths, return_logits=False)

            # 2. Tạo SOFT MASK:
            # Những từ > threshold sẽ giữ nguyên trọng số (1.0)
            # Những từ < threshold sẽ được nhân với soft_coeff (ví dụ 0.15) thay vì 0
            selection_mask = torch.where(
                selection_probs > self.threshold,
                torch.ones_like(selection_probs),
                torch.full_like(selection_probs, self.soft_coeff)
            )

            self.last_selection_rate = (selection_probs > self.threshold).float().mean().item()

            original_forward = model.decoder.forward

            def masked_forward(input_embeddings, hidden_states, kwargs=None):
                v_dist, h_state, n_kwargs = original_forward(input_embeddings, hidden_states, kwargs)

                if 'attn_dists' in n_kwargs:
                    curr_attn = n_kwargs['attn_dists'] # (B, 1, L)

                    # 3. Áp dụng Soft Mask lên Attention
                    m_attn = curr_attn * selection_mask.unsqueeze(1)

                    # 4. Re-normalize: Đảm bảo tổng attention vẫn bằng 1
                    m_sum = m_attn.sum(dim=2, keepdim=True)
                    m_attn = m_attn / (m_sum + 1e-9)

                    n_kwargs['attn_dists'] = m_attn

                    # Tính toán lại phân phối từ vựng cuối cùng với Attention mới
                    if hasattr(model, 'calculate_final_dist'):
                        v_dist = model.calculate_final_dist(ext_idx, v_dist, m_attn, n_kwargs.get('p_gens'), extra_zeros)

                return v_dist, h_state, n_kwargs

            model.decoder.forward = masked_forward
            gen_ids = model.predict(input_ids, ext_idx, extra_zeros)
            model.decoder.forward = original_forward
            return gen_ids

    def decode(self, indices, oov_list_batch):
        res = []
        for i, idx_list in enumerate(indices):
            words = []
            for idx in idx_list:
                idx = idx.item()
                if idx < self.vocab.vocab_size:
                    w = self.vocab.itos[idx]
                    if w == self.vocab.eos_token: break
                    if w not in [self.vocab.pad_token, self.vocab.bos_token]: words.append(w)
                else:
                    oov_idx = idx - self.vocab.vocab_size
                    if oov_idx < len(oov_list_batch[i]):
                        words.append(oov_list_batch[i][oov_idx])
            res.append(" ".join(words))
        return res

    def run(self, output_path):
        results, gens, gts = {}, {}, {}
        rates = []
        print(f"Starting Bottom-Up Inference (Soft Mask Coeff: {self.soft_coeff})")
        for i, items in enumerate(tqdm(self.test_dataloader)):
            input_ids = items.input_ids.to(self.device)
            ext_idx = items.extended_source_idx.to(self.device)
            extra_zeros = items.extra_zeros.to(self.device)
            lengths = torch.ne(input_ids, self.vocab.pad_idx).sum(dim=1).long()

            pred_ids = self.apply_bottom_up_mask(self.pointer_generator, input_ids, lengths, ext_idx, extra_zeros)
            p, t = self.decode(pred_ids, items.oov_list)[0], self.decode(items.label, items.oov_list)[0]

            sid = items.id[0]
            gens[sid], gts[sid] = p, t
            results[sid] = {"pred": p, "gt": t, "rate": self.last_selection_rate}
            rates.append(self.last_selection_rate)

        print(f"\nAvg Selection Rate: {sum(rates)/len(rates):.2%}")
        scores = evaluation.compute_scores(gts, gens)
        print("ROUGE Scores:", scores)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)