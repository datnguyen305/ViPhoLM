from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import json
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn
import evaluation # change it pls 

@META_TASK.register()
class MachineTranslationTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score # change it pls
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
            batch_size=128,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
    
    def get_vocab(self): 
        return self.vocab

    def train(self):
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % (self.epoch+1), unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                # forward pass
                input_vietnamese = items.input_vietnamese
                input_english = items.input_english

                # input_vietnamese: (B, S_viet, 3); Ex: [(<bos>, <pad>, <pad>), (<initiate>, <rhyme>, <tone>), (<eos>, <pad>, <pad>)]
                # input_english: (B, S_eng)

                _, loss = self.model(input_english, input_vietnamese)
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                running_loss += loss.item()

                # update the training status
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        gens = {}
        gts = {}
        processed_count = 0
        max_test_samples = 50 

        with tqdm(desc='Evaluating', unit='it', total=min(len(dataloader), 5)) as pbar:
            for items in dataloader:
                if processed_count >= max_test_samples:
                    break
                    
                items = items.to(self.device)
                with torch.no_grad():
                    # Model dự đoán
                    prediction = self.model.predict(items.input_english)
                    
                    # Giải mã toàn bộ batch
                    decoded_preds = self.vocab.decode_batch_caption_vietnamese(prediction)
                    # Lấy nhãn gốc tiếng Việt để tính BLEU chính xác
                    decoded_labels = self.vocab.decode_batch_caption_vietnamese(items.input_vietnamese)

                    # --- PHẦN IN DEBUG: CHỈ IN 1 CÂU DUY NHẤT CỦA CẢ QUÁ TRÌNH ---
                    if processed_count == 0:
                        tqdm.write("\n" + "="*30)
                        tqdm.write(f"SAMPLE DEBUG:")
                        tqdm.write(f"  - EN: {self.vocab.decode_sentence_english(items.input_english[0:1])[0]}")
                        tqdm.write(f"  - GT: {decoded_labels[0]}")
                        tqdm.write(f"  - PD: {decoded_preds[0]}")
                        tqdm.write("="*30 + "\n")

                    # Lưu 50 câu vào dict để tính điểm
                    for i in range(len(items.id)):
                        if processed_count >= max_test_samples: break
                        idx = items.id[i]
                        gens[idx] = decoded_preds[i]
                        gts[idx] = decoded_labels[i]
                        processed_count += 1

                pbar.update()
    
        # Tính BLEU trên 50 câu
        scores = evaluation.compute_bleu_scores(gts, gens)
        return scores, (gens, gts)

    def get_predictions(self):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            self.logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

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
