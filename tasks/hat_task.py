from builders.task_builder import META_TASK
from tasks.base_task import BaseTask
from torch.utils.data import DataLoader
from data_utils import collate_fn
import torch
from tqdm import tqdm
import os
import json
import torch.nn.functional as F
from builders.dataset_builder import build_dataset
import evaluation  

@META_TASK.register()
class HATClassificationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup
        self.num_labels = config.model.num_labels

    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.train, self.vocab)
        self.dev_dataset = build_dataset(config.dev, self.vocab)
        self.test_dataset = build_dataset(config.test, self.vocab)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )

    def train(self):
        self.model.train()
        total_loss = 0
        for items in self.train_dataloader:
            items = items.to(self.device)
            input_ids = items.input_ids
            labels = items.label  # Assume integer labels: [B]
            
            logits, loss = self.model(input_ids, labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

def evaluate_metrics(self, dataloader: DataLoader):
    self.model.eval()
    preds, gts = {}, {}
    with tqdm(desc=f"Epoch {self.epoch+1} - Evaluating", unit='it', total=len(dataloader)) as pbar:
        for items in dataloader:
            items = items.to(self.device)
            input_ids = items.input_ids
            label = items.label
            with torch.no_grad():
                logits = self.model.predict(input_ids)
                prediction = torch.argmax(logits, dim=-1)

                # Nếu bạn muốn giữ dạng text:
                pred_text = self.vocab.decode_sentence(prediction)
                label_text = self.vocab.decode_sentence(label)

                id = items.id[0]
                preds[id] = pred_text[0]
                gts[id] = label_text[0]

            pbar.update()

    self.logger.info("Getting scores")
    scores = evaluation.compute_scores(gts, preds)  # có thể là accuracy, f1, rouge, v.v.
    return scores, (preds, gts)

def get_predictions(self):
    ckpt = os.path.join(self.checkpoint_path, "best_model.pth")
    if not os.path.isfile(ckpt):
        self.logger.error("Model not trained. No checkpoint found!")
        raise FileNotFoundError(f"Missing file: {ckpt}")

    self.load_checkpoint(ckpt)
    self.model.eval()

    scores, (preds, gts) = self.evaluate_metrics(self.test_dataloader)

    results = {}
    with tqdm(desc=f"Epoch {self.epoch+1} - Getting results", unit='it', total=len(preds)) as pbar:
        for id in preds:
            results[id] = {
                "prediction": preds[id],
                "target": gts[id]
            }
            pbar.update()

    self.logger.info("Test scores %s", scores)
    json.dump(scores, open(os.path.join(self.checkpoint_path, "scores.json"), "w+"), ensure_ascii=False, indent=4)
    json.dump(results, open(os.path.join(self.checkpoint_path, "predictions.json"), "w+"), ensure_ascii=False, indent=4)

