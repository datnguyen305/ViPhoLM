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
        self.threshold = cfg.inference.threshold  # fallback only
        self.soft_coeff = cfg.inference.soft_coeff
        self.coverage_beta = cfg.inference.coverage_beta

        # ---- Vocab
        self.logger.info("Building vocabulary...")
        self.vocab = build_vocab(self.pg_config.vocab)

        # ---- Load models
        self.content_selector, cs_ckpt = self._load_model(
            self.cs_config,
            cfg.content_selector.checkpoint,
            "Content Selector"
        )

        # ---- Load optimized threshold from CS checkpoint
        if isinstance(cs_ckpt, dict) and "pred_threshold" in cs_ckpt:
            self.threshold = cs_ckpt["pred_threshold"]
            self.logger.info(
                f"✓ Loaded optimized CS threshold = {self.threshold:.2f}"
            )
        else:
            self.logger.warning(
                "⚠ CS checkpoint has no pred_threshold, using config threshold"
            )

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

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{name} checkpoint not found: {ckpt_path}")

        self.logger.info(f"-> Loading {name} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        state = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)

        return model.to(self.device).eval(), ckpt

    # =========================
    # Bottom-Up Mask + Coverage
    # =========================
    def apply_bottom_up_mask(self, model, input_ids, lengths, ext_idx, extra_zeros):
        with torch.no_grad():
            # ---- Content Selector
            cs_input = input_ids.clone()
            cs_input[cs_input >= self.vocab.vocab_size] = self.vocab.unk_idx

            probs = self.content_selector(cs_input, lengths, return_logits=False)

            valid_mask = (input_ids != self.vocab.pad_idx).float()

            hard_mask = (probs > self.threshold).float()
            mask = hard_mask + (1.0 - hard_mask) * self.soft_coeff
            mask = mask * valid_mask

            self.last_selection_rate = (
                hard_mask * valid_mask
            ).sum().item() / valid_mask.sum().item()

            original_forward = model.decoder.forward
            coverage = None

            def masked_forward(emb, hidden, kwargs=None):
                nonlocal coverage

                v_dist, h, kw = original_forward(emb, hidden, kwargs)

                if "attn_dists" in kw:
                    attn = kw["attn_dists"]  # [B, 1, T]

                    # ---- Bottom-up mask
                    attn = attn * mask.unsqueeze(1)
                    attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-9)

                    # ---- Coverage
                    if coverage is None:
                        coverage = attn
                    else:
                        coverage = coverage + attn

                    if self.coverage_beta > 0:
                        cov_pen = torch.min(attn, coverage).sum(dim=2)
                        attn = attn * torch.exp(
                            -self.coverage_beta * cov_pen
                        ).unsqueeze(2)
                        attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-9)

                    kw["attn_dists"] = attn

                    if hasattr(model, "calculate_final_dist"):
                        v_dist = model.calculate_final_dist(
                            ext_idx, v_dist, attn, kw.get("p_gens"), extra_zeros
                        )

                return v_dist, h, kw

            try:
                model.decoder.forward = masked_forward
                preds = model.predict(input_ids, ext_idx, extra_zeros)
            finally:
                model.decoder.forward = original_forward

            return preds

    # =========================
    # Decode
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

            pred_ids = self.apply_bottom_up_mask(
                self.pointer_generator,
                input_ids,
                lengths,
                ext_idx,
                extra_zeros
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
