import os
import torch
import pickle
import json
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from builders.vocab_builder import build_vocab
from utils.logging_utils import setup_logger
from data_utils import collate_fn
from models.bottom_up.content_selector import ContentSelector, create_selection_labels

@META_TASK.register()
class ContentSelectorTask:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()

        self.checkpoint_path = os.path.join(
            config.training.checkpoint_path,
            config.model.name
        )
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # --- VOCAB ---
        vocab_path = os.path.join(self.checkpoint_path, "vocab.bin")
        if os.path.exists(vocab_path):
            self.logger.info("Loading vocab from %s", vocab_path)
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            self.logger.info("Creating vocab")
            self.vocab = self.load_vocab(config.vocab)
            self.logger.info("Saving vocab to %s", vocab_path)
            with open(vocab_path, "wb") as f:
                pickle.dump(self.vocab, f)

        # --- DATA ---
        self.train_dataset = build_dataset(config.dataset.train, self.vocab)
        self.dev_dataset = build_dataset(config.dataset.dev, self.vocab)
        self.test_dataset = build_dataset(config.dataset.test, self.vocab)

        common_dl_params = {
            "batch_size": config.dataset.batch_size,
            "num_workers": config.dataset.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": True
        }
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **common_dl_params)
        self.dev_loader = DataLoader(self.dev_dataset, shuffle=False, **common_dl_params)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **common_dl_params)

        # --- MODEL ---
        self.device = torch.device(config.model.device)
        self.model = ContentSelector(config.model, self.vocab).to(self.device)

        # --- OPTIMIZER & LOSS ---
        self.optimizer = Adam(self.model.parameters(), lr=config.training.learning_rate)
        pos_weight = torch.tensor([config.training.pos_weight], device=self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        
        # Scheduler theo dõi F1 trên Dev
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=2, factor=0.5
        )

        # --- STATE ---
        self.epoch = 0
        self.best_f1 = 0.0
        self.pred_threshold = config.training.pred_threshold
        self.target_ratio = config.training.target_ratio
        self.patience_limit = config.training.patience
        
        self._load_checkpoint()

    def find_optimal_threshold(self, all_probs, all_labels):
        """Tìm ngưỡng tối ưu để đạt F1-score cao nhất trên tập Validation"""
        best_threshold = 0.5
        max_f1 = 0
        # Thử nghiệm các ngưỡng từ 0.1 đến 0.9
        thresholds = np.linspace(0.1, 0.9, 17)
        for t in thresholds:
            preds = (all_probs > t).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > max_f1:
                max_f1 = f1
                best_threshold = t
        return float(best_threshold), float(max_f1)
    
    def load_vocab(self, config):
        vocab = build_vocab(config)

        return vocab

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")

        for batch in pbar:
            batch = batch.to(self.device)
            src, tgt = batch.input_ids, batch.label
            src_lengths = (src != self.vocab.pad_idx).sum(dim=1)

            labels = create_selection_labels(src, tgt, self.vocab, target_ratio=self.target_ratio)
            logits = self.model(src, src_lengths, return_logits=True)

            mask = (src != self.vocab.pad_idx).float()
            loss = (self.criterion(logits, labels) * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self, dataloader, split_name="Dev", search_threshold=False):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {self.epoch + 1} [{split_name}]"):
                batch = batch.to(self.device)
                src, tgt = batch.input_ids, batch.label
                src_lengths = (src != self.vocab.pad_idx).sum(dim=1)

                labels = create_selection_labels(src, tgt, self.vocab, target_ratio=self.target_ratio)
                logits = self.model(src, src_lengths, return_logits=True)

                mask = (src != self.vocab.pad_idx).float()
                loss = (self.criterion(logits, labels) * mask).sum() / mask.sum()
                total_loss += loss.item()

                probs = torch.sigmoid(logits)
                for b in range(src.size(0)):
                    L = src_lengths[b]
                    all_probs.extend(probs[b, :L].cpu().tolist())
                    all_labels.extend(labels[b, :L].cpu().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Nếu là tập Dev, ta tìm ngưỡng tối ưu
        if search_threshold:
            new_threshold, _ = self.find_optimal_threshold(all_probs, all_labels)
            self.pred_threshold = new_threshold
            self.logger.info(f"--- Optimized Threshold to: {self.pred_threshold:.2f} ---")

        preds = (all_probs > self.pred_threshold).astype(int)
        
        metrics = {
            "loss": total_loss / len(dataloader),
            "f1": f1_score(all_labels, preds, zero_division=0),
            "precision": precision_score(all_labels, preds, zero_division=0),
            "recall": recall_score(all_labels, preds, zero_division=0),
            "selection_rate": np.mean(preds),
            "gold_rate": np.mean(all_labels)
        }
        
        self.logger.info(
            f"{split_name} - F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | "
            f"R: {metrics['recall']:.4f} | Sel Rate: {metrics['selection_rate']:.3f}"
        )
        return metrics

    def start(self):
        patience = 0

        while True:
            self.logger.info(f"Epoch {self.epoch + 1} started")

            # ---- TRAIN ----
            self.train_one_epoch()

            # ---- DEV (optimize threshold) ----
            dev_metrics = self.evaluate(
                self.dev_loader,
                "Dev",
                search_threshold=True
            )
            dev_f1 = dev_metrics["f1"]

            # ---- SCHEDULER ----
            self.scheduler.step(dev_f1)

            # ---- EARLY STOPPING ----
            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
                patience = 0

                self._save_checkpoint(is_best=True)
                self.logger.info(
                    f"✓ New BEST model | "
                    f"F1={self.best_f1:.4f} | "
                    f"Threshold={self.pred_threshold:.2f}"
                )
            else:
                patience += 1
                self.logger.info(
                    f"No improvement. Patience {patience}/{self.patience_limit}"
                )

                if patience >= self.patience_limit:
                    self.logger.info("Early stopping triggered")
                    break

            # ---- NEXT EPOCH ----
            self.epoch += 1
            self._save_checkpoint(is_best=False)

        # ---- TEST ----
        self.test()


    def test(self):
        self.logger.info("Evaluating on Test Set...")
        best_path = os.path.join(self.checkpoint_path, "best_model.pth")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(state["state_dict"])
            self.pred_threshold = state.get("pred_threshold", 0.5)

        test_metrics = self.evaluate(self.test_loader, "Test")
        with open(os.path.join(self.checkpoint_path, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    def _save_checkpoint(self, is_best=False):
        state = {
            "epoch": self.epoch,
            "best_f1": self.best_f1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "pred_threshold": self.pred_threshold,
            "target_ratio": self.target_ratio,
        }
        name = "best_model.pth" if is_best else "last_model.pth"
        torch.save(state, os.path.join(self.checkpoint_path, name))

    def _load_checkpoint(self):
        path = os.path.join(self.checkpoint_path, "last_model.pth")
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.epoch = state["epoch"] + 1
            self.best_f1 = state["best_f1"]
            self.pred_threshold = state.get("pred_threshold", 0.5)
            self.logger.info(f"Resumed from epoch {self.epoch}")