import os
import json
import yaml
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from builders.vocab_builder import build_vocab
from builders.dataset_builder import build_dataset
from builders.model_builder import build_model
from data_utils import collate_fn_oov
from utils.logging_utils import setup_logger
from utils.instance import Instance
import evaluation


# =========================
# Utils
# =========================
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def _to_inst(x):
        if isinstance(x, dict):
            inst = Instance()
            for k, v in x.items():
                setattr(inst, k, _to_inst(v))
            return inst
        if isinstance(x, list):
            return [_to_inst(i) for i in x]
        return x

    return _to_inst(data)


# =========================
# Bottom-Up Inference
# =========================
class BottomUpSummarizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger()

        # ---- Load configs
        self.cs_config = load_config(cfg.content_selector.config)
        self.pg_config = load_config(cfg.pointer_generator.config)

        # ---- Inference params
        self.threshold = cfg.inference.threshold
        self.soft_coeff = cfg.inference.soft_coeff

        # ---- Vocab
        self.logger.info("Building vocabulary...")
        self.vocab = build_vocab(self.pg_config.vocab)

        # ---- Load models
        self.content_selector, cs_ckpt = self._load_model(
            self.cs_config,
            cfg.content_selector.checkpoint,
            "Content Selector"
        )

        # ---- Load optimized threshold (optional)
        if isinstance(cs_ckpt, dict) and "pred_threshold" in cs_ckpt:
            self.threshold = cs_ckpt["pred_threshold"]
            self.logger.info(f"✓ Loaded optimized CS threshold = {self.threshold:.2f}")

        self.pointer_generator, _ = self._load_model(
            self.pg_config,
            cfg.pointer_generator.checkpoint,
            "Pointer Generator"
        )

        self.logger.info(
            f"Using threshold={self.threshold:.2f}, soft_coeff={self.soft_coeff}"
        )

        # ---- Dataset
        self.test_dataset = build_dataset(self.pg_config.dataset.test, self.vocab)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            collate_fn=collate_fn_oov
        )

        self.output_dir = cfg.inference.output_dir

    def _load_model(self, config, ckpt_path, name):
        config.model.device = self.device
        model = build_model(config.model, self.vocab)

        self.logger.info(f"-> Loading {name} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        state = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)

        return model.to(self.device).eval(), ckpt

    def greedy_decode(self, model, input_ids, lengths,
                  ext_idx, extra_zeros, max_len):

        B, src_len = input_ids.size()
        device = input_ids.device
    
        # ===== 1. Content Selector =====
        cs_input = input_ids.clone()
        cs_input[cs_input >= self.vocab.vocab_size] = self.vocab.unk_idx
    
        probs = self.content_selector(cs_input, lengths, return_logits=False)
        valid_mask = (input_ids != self.vocab.pad_idx).float()
    
        hard_mask = (probs > self.threshold).float()
        selector_mask = (hard_mask + (1 - hard_mask) * self.soft_coeff) * valid_mask
    
        self.last_selection_rate = (
            (hard_mask * valid_mask).sum() / valid_mask.sum()
        ).item()
    
        # ===== 2. Encode =====
        encoder_outputs, decoder_hidden = model.encode(input_ids, lengths)
    
        # ===== 3. Decoder kwargs =====
        kwargs = {
            "encoder_outputs": encoder_outputs,
            "encoder_masks": (input_ids != self.vocab.pad_idx),
            "context": torch.zeros((B, 1, model.context_size), device=device),
            "extra_zeros": extra_zeros,
            "extended_source_idx": ext_idx,
            "selector_mask": selector_mask,
        }
    
        if model.is_coverage:
            kwargs["coverages"] = torch.zeros((B, 1, src_len), device=device)
    
        # ===== 4. Greedy Decode =====
        ys = torch.full((B, 1),
                        self.vocab.bos_idx,
                        device=device,
                        dtype=torch.long)
    
        outputs = []
    
        for t in range(max_len):
            inp = ys[:, -1:].clone()
            inp[inp >= self.vocab.vocab_size] = self.vocab.unk_idx
            inp = model.target_token_embedder(inp)
    
            final_dist, decoder_hidden, kwargs = model.decoder(
                inp, decoder_hidden, kwargs=kwargs
            )
    
            # final_dist: [B, 1, V_ext] → [B, V_ext]
            final_dist = final_dist.squeeze(1)
    
            # ===== NO-REPEAT 2-GRAM =====
            if t >= 2:
                for b in range(B):
                    last_2 = ys[b, -2:].tolist()
                    for i in range(len(ys[b]) - 2):
                        if ys[b, i:i+2].tolist() == last_2:
                            forbidden = ys[b, i+2].item()
                            if forbidden < final_dist.size(1):
                                final_dist[b, forbidden] = -1e10
    
            next_tok = final_dist.argmax(-1).unsqueeze(1) 

            outputs.append(next_tok)
            ys = torch.cat([ys, next_tok], dim=1)

    
            if (next_tok == self.vocab.eos_idx).all():
                break
    
        return torch.cat(outputs, dim=1)



    # =========================
    # Decode indices → text
    # =========================
    def decode(self, indices, oov_list):
        results = []
        for i, seq in enumerate(indices):
            words = []
            for idx in seq:
                idx = idx.item()
                if idx < self.vocab.vocab_size:
                    w = self.vocab.itos[idx]
                    if w == self.vocab.eos_token:
                        break
                    if w not in (self.vocab.pad_token, self.vocab.bos_token):
                        words.append(w)
                else:
                    oov_id = idx - self.vocab.vocab_size
                    if oov_id < len(oov_list[i]):
                        words.append(oov_list[i][oov_id])
            results.append(" ".join(words))
        return results

    # =========================
    # Run
    # =========================
    def run(self):
        gens, gts, preds = {}, {}, {}
        rates = []

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("Starting Bottom-Up Inference...")

        for items in tqdm(self.test_loader):
            input_ids = items.input_ids.to(self.device)
            ext_idx = items.extended_source_idx.to(self.device)
            extra_zeros = items.extra_zeros.to(self.device)
            lengths = (input_ids != self.vocab.pad_idx).sum(dim=1)

            pred_ids = self.greedy_decode(
                self.pointer_generator,
                input_ids,
                lengths,
                ext_idx,
                extra_zeros,
                max_len=200
            )

            pred_texts = self.decode(pred_ids, items.oov_list)
            gt_texts = self.decode(items.label, items.oov_list)

            for i, sid in enumerate(items.id):
                gens[sid] = pred_texts[i]
                gts[sid] = gt_texts[i]
                preds[sid] = {
                    "prediction": pred_texts[i],
                    "target": gt_texts[i],
                    "selection_rate": self.last_selection_rate
                }

            rates.append(self.last_selection_rate)

        self.logger.info(f"Avg Selection Rate: {sum(rates)/len(rates):.2%}")
        scores = evaluation.compute_scores(gts, gens)
        self.logger.info(f"ROUGE: {scores}")

        json.dump(
            preds,
            open(os.path.join(self.output_dir, "predictions.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4
        )
        json.dump(
            scores,
            open(os.path.join(self.output_dir, "scores.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4
        )


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    infer = BottomUpSummarizer(cfg)
    infer.run()
