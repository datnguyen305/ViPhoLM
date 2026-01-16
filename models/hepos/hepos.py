import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from builders.model_builder import META_ARCHITECTURE
from models.hepos.patch_bart import patch_bart_with_hepos


@META_ARCHITECTURE.register()
class HEPOSBartSummarizer(nn.Module):
    """
    BART + HEPOS cross-attention
    """

    def __init__(self, config, vocab=None):
        super().__init__()

        self.model = BartForConditionalGeneration.from_pretrained(
            config.pretrained_name
        )

        patch_bart_with_hepos(
            self.model,
            stride=config.stride
        )
        self.d_model = self.model.config.d_model

        self.model.to(config.device)

    def forward(self, input_ids, labels):
        for p in self.model.model.encoder.parameters():
            p.requires_grad = False
        out = self.model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
        )
        return out.logits, out.loss

    @torch.no_grad()
    def predict(
        self,
        input_ids,
        beam_size=4,
        max_len=256,
        min_len=40,
        length_penalty=2.0,
    ):
        return self.model.generate(
            input_ids=input_ids,
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
