from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import json

from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn_seneca
import evaluation


@META_TASK.register()
class TextSumTaskSeneca(BaseTask):
    """
    Text Summarization Task cho SENECA
    Giữ format giống TextSumTask gốc
    """

    def __init__(self, config):
        super().__init__(config)


    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup
        self.lambda_cov = getattr(config.training, "lambda_cov", 1.0)


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
            collate_fn=collate_fn_seneca
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn_seneca
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn_seneca
        )

    def get_vocab(self):
        return self.vocab

    def train(self):
        self.model.train()

        running_loss = 0.0
        with tqdm(
            desc='Epoch %d - Training' % (self.epoch + 1),
            unit='it',
            total=len(self.train_dataloader)
        ) as pbar:

            for it, items in enumerate(self.train_dataloader):

                sentences = items["sentences"].to(self.device)
                labels = items["label"].to(self.device)
                entities = items["entities"]


                outputs, _, _, cov_loss = self.model(
                    sentences,
                    entities,
                    target=labels,
                    return_coverage=True
                )

                vocab_size = outputs.size(-1)
                loss_nll = torch.nn.functional.nll_loss(
                    torch.log(outputs.view(-1, vocab_size) + 1e-12),
                    labels.view(-1),
                    ignore_index=self.vocab.pad_idx
                )

                loss = loss_nll + self.lambda_cov * cov_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()


                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()


    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        gens = {}
        gts = {}

        with tqdm(
            desc='Epoch %d - Evaluating' % (self.epoch + 1),
            unit='it',
            total=len(dataloader)
        ) as pbar:

            for items in dataloader:
                sentences = items["sentences"].to(self.device)
                labels = items["label"].to(self.device)
                entities = items["entities"]


                with torch.no_grad():
                    outputs, _, _ = self.model(
                        sentences,
                        entities,
                        target=None
                    )

                    preds = torch.argmax(outputs, dim=-1)

                    preds = self.vocab.decode_sentence(preds)
                    labels = self.vocab.decode_sentence(labels)

                    sample_id = items["id"][0]
                    gens[sample_id] = preds[0]
                    gts[sample_id] = labels[0]

                pbar.update()

        self.logger.info("Getting scores")
        scores = evaluation.compute_scores(gts, gens)
        return scores, (gens, gts)

    def get_predictions(self):
        ckpt = os.path.join(self.checkpoint_path, 'best_model.pth')
        if not os.path.isfile(ckpt):
            self.logger.error(
                "Prediction requires trained model (best_model.pth not found)"
            )
            raise FileNotFoundError("best_model.pth not found")

        self.load_checkpoint(ckpt)
        self.model.eval()

        scores, (gens, gts) = self.evaluate_metrics(self.test_dataloader)

        results = {}
        with tqdm(
            desc='Epoch %d - Getting results' % (self.epoch + 1),
            unit='it',
            total=len(gens)
        ) as pbar:

            for sample_id in gens:
                results[sample_id] = {
                    "prediction": gens[sample_id],
                    "target": gts[sample_id]
                }
                pbar.update()

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
