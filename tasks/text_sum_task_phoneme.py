from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import json
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn
import evaluation

@META_TASK.register()
class TextSumTaskPhoneme(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup

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
            batch_size=1, # Giữ batch=1 theo ý bạn
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1, # Giữ batch=1 theo ý bạn
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
    
    def get_vocab(self): 
        return self.vocab

    def train(self):
        self.model.train()
        train_losses = []
        running_loss = .0
        
        with tqdm(desc='Epoch %d - Training' % (self.epoch+1), unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                input_ids = items.input_ids
        
                # --- QUAN TRỌNG: SỬA LABEL ---
                # Dùng label gốc (có BOS và EOS). Model sẽ tự cắt để làm input và target.
                labels = items.label 
                
                # Forward
                _, loss = self.model(input_ids, labels)
                
                # Backward
                self.optim.zero_grad()
                loss.backward()
                
                # Clip gradient để tránh lỗi nổ gradient với model phức tạp
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optim.step()
                
                # Logging
                current_loss = loss.item()
                running_loss += current_loss
                train_losses.append(current_loss)

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()
                
        return train_losses

    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        gens = {}
        gts = {}
        
        # Biến debug
        debug_count = 0

        with torch.inference_mode():
            with tqdm(desc='Epoch %d - Evaluating' % (self.epoch+1), unit='it', total=len(dataloader)) as pbar:
                for items in dataloader:
                    items = items.to(self.device)
                    input_ids = items.input_ids
                    
                    # Predict: [Batch, Seq, 4]
                    prediction = self.model.predict(input_ids)

                    # --- QUAN TRỌNG: SỬA DECODE ---
                    # Dùng decode_batch_caption của ViWordVocab để xử lý output 4 thành phần
                    decoded_preds = self.vocab.decode_batch_caption(prediction, join_words=True)
                    decoded_labels = self.vocab.decode_batch_caption(items.label, join_words=True)

                    # Debug: In thử 1 câu ra xem model chạy thế nào
                    if debug_count < 1:
                        print(f"\n[DEBUG Sample] GT: {decoded_labels[0]}")
                        print(f"[DEBUG Sample] PR: {decoded_preds[0]}")
                        debug_count += 1

                    # Lưu kết quả
                    uid = items.id[0]
                    gens[uid] = decoded_preds[0]
                    gts[uid] = decoded_labels[0]

                    pbar.update()
        
        # Calculate metrics (ROUGE, BLEU, etc.)
        self.logger.info("Getting scores")
        scores = evaluation.compute_scores(gts, gens)
    
        return scores, (gens, gts)

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            self.logger.error("Prediction requires a trained model.")
            raise FileNotFoundError("best_model.pth not found")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        scores, (gens, gts) = self.evaluate_metrics(self.test_dataloader)
        results = {}
        with tqdm(desc='Epoch %d - Getting results' % (self.epoch+1), unit='it', total=len(gens)) as pbar:
            for id in gens:
                gen = gens[id]
                gt = gts[id]
                results[id] = {
                    "prediction": gen,
                    "target": gt
                }
                
                pbar.update()

        self.logger.info("Test scores %s", scores)
        json.dump(scores, open(os.path.join(self.checkpoint_path, "scores.json"), "w+"), ensure_ascii=False, indent=4)
        json.dump(results, open(os.path.join(self.checkpoint_path, "predictions.json"), "w+"), ensure_ascii=False, indent=4)