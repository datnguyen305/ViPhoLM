from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import json

from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn_viword as collate_fn
import evaluation


@META_TASK.register()
class TextSumTaskViWord(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    # =========================
    # Hyperparameters
    # =========================
    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup

    # =========================
    # Dataset
    # =========================
    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.train, self.vocab)
        self.dev_dataset   = build_dataset(config.dev, self.vocab)
        self.test_dataset  = build_dataset(config.test, self.vocab)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )

        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )

    def get_vocab(self):
        return self.vocab

    # =========================
    # Utils
    # =========================
    def _move_to_device(self, batch):
        """Move tensor fields in dict batch to device"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    # =========================
    # Train
    # =========================
    def train(self):
        self.model.train()
        running_loss = 0.0

        with tqdm(
            desc=f"Epoch {self.epoch + 1} - Training",
            total=len(self.train_dataloader),
            unit="it"
        ) as pbar:

            for it, batch in enumerate(self.train_dataloader):
                batch = self._move_to_device(batch)

                src = batch["input_ids"]              # (B, N, S)
                trg = batch["shifted_right_label"]    # (B, T)

                # Model.forward expects (src, trg) where trg includes <bos> and <eos>
                _, loss = self.model(src, trg)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    # =========================
    # Evaluation
    # =========================
    def evaluate_metrics(self, dataloader):
        self.model.eval()
        gens, gts = {}, {}

        with tqdm(
            desc=f"Epoch {self.epoch + 1} - Evaluating",
            total=len(dataloader),
            unit="it"
        ) as pbar:

            for batch in dataloader:
                batch = self._move_to_device(batch)

                src = batch["input_ids"]              # (B, N, S)
                label = batch["shifted_right_label"]  # (B, T) - full target with <bos> and <eos>

                with torch.no_grad():
                    pred = self.model.predict(src)    # (B, T_pred)

                # Decode predictions and labels
                pred_text = self.vocab.decode_sentence(pred)
                label_text = self.vocab.decode_sentence(label)

                sample_id = batch["id"][0]
                gens[sample_id] = pred_text[0]
                gts[sample_id] = label_text[0]

                pbar.update()

        self.logger.info("Getting scores")
        scores = evaluation.compute_scores(gts, gens)
        return scores, (gens, gts)

    # =========================
    # Prediction
    # =========================
    def get_predictions(self):
        best_ckpt = os.path.join(self.checkpoint_path, "best_model.pth")
        if not os.path.isfile(best_ckpt):
            raise FileNotFoundError("best_model.pth not found")

        self.load_checkpoint(best_ckpt)
        self.model.eval()

        scores, (gens, gts) = self.evaluate_metrics(self.test_dataloader)

        results = {
            k: {"prediction": gens[k], "target": gts[k]}
            for k in gens
        }

        self.logger.info("Test scores %s", scores)

        json.dump(
            scores,
            open(os.path.join(self.checkpoint_path, "scores.json"), "w"),
            ensure_ascii=False,
            indent=4
        )

        json.dump(
            results,
            open(os.path.join(self.checkpoint_path, "predictions.json"), "w"),
            ensure_ascii=False,
            indent=4
        )