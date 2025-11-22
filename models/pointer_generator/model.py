import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random

from builders.model_builder import META_ARCHITECTURE
from models.pointer_generator.layers import Encoder, Decoder, ReduceState
from vocabs.vocab import Vocab
from utils.get_oov import get_oovs, article_oovs_to_extended_vocab

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


@META_ARCHITECTURE.register()
class PointerGeneratorModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.d_model = config.d_model
        self.MAX_LENGTH = vocab.max_sentence_length + 2

        # Encoder, Decoder, ReduceState
        self.encoder = Encoder(config.encoder, vocab)
        self.decoder = Decoder(config.decoder, vocab)
        self.reduce_state = ReduceState(config.reduce_state)

        # Share embedding between encoder and decoder
        self.decoder.tgt_word_emb.weight = self.encoder.src_word_emb.weight

        # Move to GPU if needed
        if config.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.reduce_state = self.reduce_state.cuda()

        # Set eval mode if needed
        if config.is_eval:
            self.encoder.eval()
            self.decoder.eval()
            self.reduce_state.eval()

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        config = self.config
        vocab = self.vocab
        batch_size = x.size(0)

        # Lengths & sort. Sắp xếp batch theo độ dài giảm dần để RNN xử lý hiệu quả hơn.
        x_lens = (x != vocab.pad_idx).sum(dim=1)
        x_lens_sorted, sort_indices = torch.sort(x_lens, descending=True)
        x_sorted = x[sort_indices]
        labels_sorted = labels[sort_indices]
        _, unsort_indices = torch.sort(sort_indices)

        # Pointer-generator: get OOVs and extend vocab
        enc_batch_extend_vocab = x_sorted.clone()
        extra_zeros = None
        batch_oovs = None
        
        if config.pointer_gen:
            batch_oovs, max_oov_size = get_oovs(x_sorted.tolist(), vocab)
            config.max_oov_size = max_oov_size
            
            # Create extended vocab indices for encoder input
            enc_batch_extend_vocab = article_oovs_to_extended_vocab(
                x_sorted.tolist(), batch_oovs, vocab
            )
            enc_batch_extend_vocab = torch.tensor(
                enc_batch_extend_vocab, dtype=torch.long, device=x.device
            )
            
            if max_oov_size > 0:
                extra_zeros = torch.zeros(
                    (batch_size, max_oov_size), dtype=torch.float, device=x.device
                )

        # Encode
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(x_sorted, x_lens_sorted)
        decoder_hidden = self.reduce_state(encoder_hidden)
        c_t = torch.zeros(batch_size, config.hidden_dim * 2, device=x.device)
        coverage = torch.zeros(batch_size, x_sorted.size(1), device=x.device) if config.is_coverage else None

        # Decoder input start with BOS
        decoder_input = torch.full((batch_size, 1), vocab.bos_idx, dtype=torch.long, device=x.device)

        step_losses = []
        max_dec_len = labels_sorted.size(1)

        for t in range(max_dec_len):
            final_dist, decoder_hidden, c_t, attn_dist, p_gen, coverage = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_feature,
                (x_sorted != vocab.pad_idx).float(), c_t, extra_zeros, enc_batch_extend_vocab,
                coverage, t
            )

            # Get target with extended vocab indices
            target = labels_sorted[:, t]
            if config.pointer_gen and batch_oovs is not None:
                # Convert target to extended vocab if needed
                target_extended = []
                for b in range(batch_size):
                    tgt = target[b].item()
                    if tgt >= len(vocab):  # OOV token
                        # Find position in batch_oovs
                        if tgt in batch_oovs[b]:
                            oov_idx = batch_oovs[b].index(tgt)
                            target_extended.append(len(vocab) + oov_idx)
                        else:
                            target_extended.append(vocab.unk_idx)
                    else:
                        target_extended.append(tgt)
                target = torch.tensor(target_extended, dtype=torch.long, device=x.device)

            # tính loss
            log_probs = torch.log(final_dist + 1e-12)
            step_loss = F.nll_loss(log_probs, target, ignore_index=vocab.pad_idx, reduction='sum')
            step_losses.append(step_loss)

            # Teacher forcing: use gold target
            decoder_input = labels_sorted[:, t].unsqueeze(1)

        total_loss = sum(step_losses)
        num_non_pad = (labels_sorted != vocab.pad_idx).sum().float()
        avg_loss = total_loss / (num_non_pad + 1e-12)

        if config.is_coverage and coverage is not None:
            # Coverage loss: sum of minimum between coverage and attention at each position
            coverage_loss = torch.sum(torch.min(coverage, attn_dist), dim=1).sum() / batch_size
            avg_loss = avg_loss + config.cov_loss_wt * coverage_loss

        return None, avg_loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        config = self.config
        vocab = self.vocab
        batch_size = x.size(0)

        # Lengths & sort
        x_lens = (x != vocab.pad_idx).sum(dim=1)
        x_lens_sorted, sort_indices = torch.sort(x_lens, descending=True)
        x_sorted = x[sort_indices]
        _, unsort_indices = torch.sort(sort_indices)

        enc_padding_mask = (x_sorted != vocab.pad_idx).float()
        enc_batch_extend_vocab = x_sorted.clone()
        extra_zeros = None
        batch_oovs = None

        # Pointer-generator OOVs
        if config.pointer_gen:
            batch_oovs, max_oov_size = get_oovs(x_sorted.tolist(), vocab)
            config.max_oov_size = max_oov_size
            
            # Create extended vocab indices
            enc_batch_extend_vocab = article_oovs_to_extended_vocab(
                x_sorted.tolist(), batch_oovs, vocab
            )
            enc_batch_extend_vocab = torch.tensor(
                enc_batch_extend_vocab, dtype=torch.long, device=x.device
            )
            
            if max_oov_size > 0:
                extra_zeros = torch.zeros(
                    (batch_size, max_oov_size), dtype=torch.float, device=x.device
                )

        # Encode & Reduce state
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(x_sorted, x_lens_sorted)
        decoder_hidden = self.reduce_state(encoder_hidden)
        c_t = torch.zeros(batch_size, 2 * config.hidden_dim, device=x.device)
        coverage = torch.zeros(batch_size, x_sorted.size(1), device=x.device) if config.is_coverage else None
        decoder_input = torch.full((batch_size, 1), vocab.bos_idx, dtype=torch.long, device=x.device)

        outputs = []
        for t in range(self.MAX_LENGTH):
            final_dist, decoder_hidden, c_t, attn_dist, p_gen, coverage = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_feature,
                enc_padding_mask, c_t, extra_zeros, enc_batch_extend_vocab, coverage, t
            )
            
            top_idx = final_dist.argmax(dim=-1)
            
            # Convert extended vocab indices back to regular vocab for input
            decoder_input_idx = top_idx.clone()
            if config.pointer_gen:
                # If index is beyond vocab, map to UNK for embedding lookup
                decoder_input_idx = torch.where(
                    decoder_input_idx >= len(vocab),
                    torch.tensor(vocab.unk_idx, device=x.device),
                    decoder_input_idx
                )
            
            outputs.append(top_idx)
            decoder_input = decoder_input_idx.unsqueeze(1)
            
            if (top_idx == vocab.eos_idx).all():
                break

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs[unsort_indices]
        return outputs