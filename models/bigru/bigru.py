import torch
import torch.nn as nn
import torch.nn.functional as F
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE



class LuongAttention(nn.Module):
    """
    score(h_t, h_s) = h_t^T W h_s
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # [B, H]
        encoder_outputs: torch.Tensor,  # [B, L, H]
        mask: torch.Tensor | None = None
    ):
        # [B, 1, H]
        query = self.linear(decoder_hidden).unsqueeze(1)

        # energy: [B, 1, L]
        energy = torch.bmm(query, encoder_outputs.transpose(1, 2))

        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        # attention weights: [B, 1, L]
        attn_weights = torch.softmax(energy, dim=-1)

        # context vector: [B, 1, H]
        context = torch.bmm(attn_weights, encoder_outputs)

        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.hidden_size,
            padding_idx=vocab.pad_idx
        )

        self.bigru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.layer_dim,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.layer_dim > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)
        self.hidden_size = config.hidden_size
        self.bidirectional = config.bidirectional

        if self.bidirectional:
            # project 2H -> H
            self.output_proj = nn.Linear(
                config.hidden_size * 2,
                config.hidden_size
            )

    def forward(self, x: torch.Tensor):
        """
        x: [B, L]
        """
        emb = self.dropout(self.embedding(x))
        outputs, hidden = self.bigru(emb)

        if self.bidirectional:
            # outputs: [B, L, 2H] -> [B, L, H]
            outputs = self.output_proj(outputs)

            # hidden: [2*num_layers, B, H] -> [num_layers, B, H]
            hidden = self._combine_bidirectional_hidden(hidden)

        return outputs, hidden

    @staticmethod
    def _combine_bidirectional_hidden(hidden: torch.Tensor):
        """
        hidden: [num_layers*2, B, H]
        return: [num_layers, B, H]
        """
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, hidden.size(1), hidden.size(2))
        return hidden.sum(dim=1)


class AttnDecoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(
            vocab.vocab_size,
            config.hidden_size,
            padding_idx=vocab.pad_idx
        )

        self.attention = LuongAttention(config.hidden_size)

        self.gru = nn.GRU(
            input_size=config.hidden_size * 2,  # embedding + context
            hidden_size=config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout if config.layer_dim > 1 else 0.0
        )

        self.dropout = nn.Dropout(config.dropout)

        self.out = nn.Linear(
            config.hidden_size * 2,
            vocab.vocab_size
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        targets: [B, T]
        """
        batch_size, tgt_len = targets.size()
        device = encoder_outputs.device

        decoder_input = torch.full(
            (batch_size, 1),
            self.vocab.bos_idx,
            device=device,
            dtype=torch.long
        )

        hidden = encoder_hidden
        outputs = []

        for t in range(tgt_len):
            output, hidden = self.forward_step(
                decoder_input,
                hidden,
                encoder_outputs
            )
            outputs.append(output)
            decoder_input = targets[:, t].unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

    def forward_step(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ):
        # embedding
        emb = self.dropout(self.embedding(input))  # [B,1,H]

        # attention
        context, _ = self.attention(hidden[-1], encoder_outputs)  # [B,1,H]

        # GRU
        gru_input = torch.cat([emb, context], dim=-1)
        output, hidden = self.gru(gru_input, hidden)

        # output projection
        output = torch.cat([output, context], dim=-1)
        logits = self.out(output)  # [B,1,V]

        return logits, hidden


@META_ARCHITECTURE.register()
class BiGRU_Model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.d_model = config.d_model
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2

        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = AttnDecoder(config.decoder, vocab)

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_idx,
            label_smoothing=getattr(config, "label_smoothing", 0.0)
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        enc_out, enc_hidden = self.encoder(x)
        dec_out, _ = self.decoder(enc_out, enc_hidden, labels)

        loss = self.loss_fn(
            dec_out.reshape(-1, self.vocab.vocab_size),
            labels.reshape(-1)
        )

        return dec_out, loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        enc_out, enc_hidden = self.encoder(x)

        batch_size = x.size(0)
        device = x.device

        decoder_input = torch.full(
            (batch_size, 1),
            self.vocab.bos_idx,
            device=device,
            dtype=torch.long
        )

        hidden = enc_hidden
        outputs = []

        for _ in range(self.MAX_LENGTH):
            logits, hidden = self.decoder.forward_step(
                decoder_input,
                hidden,
                enc_out
            )

            decoder_input = logits.argmax(dim=-1)
            outputs.append(decoder_input)

            if (decoder_input == self.vocab.eos_idx).all():
                break

        return torch.cat(outputs, dim=1)
