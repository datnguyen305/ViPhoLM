# longformer_encoder_decoder.py
import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict
from transformers import PretrainedConfig

from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

# Import the sliding chunks functions (assuming they exist in your codebase)
from .sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from .diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations


class LongformerEncoderDecoderConfig:
    def __init__(self, config):
        """
        Args:
            attention_window: list of attention window sizes of length = number of layers.
                window size = number of attention locations on each side.
                For an effective window size of 512, use `attention_window=[256]*num_layers`
                which is 256 on each side.
            attention_dilation: list of attention dilation of length = number of layers.
                attention dilation of `1` means no dilation.
            autoregressive: do autoregressive attention or have attention of both sides
            attention_mode: 'tvm' for TVM implementation of Longformer selfattention, 
                'sliding_chunks' for another implementation of Longformer selfattention
        """
        self.d_model = config.d_model
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.attention_window = config.attention_window or [256] * config.num_hidden_layers
        self.attention_dilation = config.attention_dilation or [1] * config.num_hidden_layers
        self.autoregressive = config.autoregressive
        self.attention_mode = config.attention_mode
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.max_position_embeddings = config.max_position_embeddings
        self.device = config.device
        
        self.initializer_range = config.initializer_range
        
        # Validate attention mode
        assert self.attention_mode in ['tvm', 'sliding_chunks'], f"attention_mode must be 'tvm' or 'sliding_chunks', got {self.attention_mode}"
        
        # Additional validation for sliding_chunks
        if self.attention_mode == 'sliding_chunks':
            assert not self.autoregressive, "autoregressive not supported for sliding_chunks mode"
            assert all(d == 1 for d in self.attention_dilation), "dilation not supported for sliding_chunks mode"
        


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        if config.d_model % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.d_model, config.num_attention_heads))
        
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.d_model / config.num_attention_heads)
        self.embed_dim = config.d_model

        self.query = nn.Linear(config.d_model, self.embed_dim)
        self.key = nn.Linear(config.d_model, self.embed_dim)
        self.value = nn.Linear(config.d_model, self.embed_dim)

        self.query_global = nn.Linear(config.d_model, self.embed_dim)
        self.key_global = nn.Linear(config.d_model, self.embed_dim)
        self.value_global = nn.Linear(config.d_model, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks']
        
        if self.attention_mode == 'sliding_chunks':
            assert not self.autoregressive  # not supported for sliding_chunks
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported and should be None"
        assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported and should be None"

        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim
        
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Use selected attention mode
        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise ValueError(f"Unsupported attention mode: {self.attention_mode}")
        
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        
        if remove_from_windowed_attention_mask is not None:
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0, False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            else:
                raise ValueError(f"Unsupported attention mode: {self.attention_mode}")
            attn_weights += d_mask

        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]

        # Handle extra attention (global attention)
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        
        if key_padding_mask is not None:
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            attn = torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise ValueError(f"Unsupported attention mode: {self.attention_mode}")
        
        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # Handle global attention recomputation
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[extra_attention_mask_nonzeros[::-1]]

            q = self.query_global(selected_hidden_states)
            k = self.key_global(hidden_states)
            v = self.value_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        
        if output_attentions:
            if extra_attention_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                attn_weights = attn_weights.permute(0, 2, 1, 3)
        
        outputs = (context_layer, attn_weights) if output_attentions else (context_layer,)
        return outputs


class LongformerEncoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LongformerSelfAttention(config, layer_id)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.hidden_dropout_prob
        
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        residual = hidden_states
        
        # Get attention outputs - handle different return formats
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        if output_attentions:
            hidden_states, attn_weights = attn_outputs
        else:
            hidden_states = attn_outputs[0]  # Get the first (and only) element from tuple
            attn_weights = None
        
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states


class LongformerEncoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model
        
        self.embeddings = nn.Embedding(vocab.vocab_size, self.embed_dim, padding_idx=vocab.pad_idx)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([
            LongformerEncoderLayer(config, layer_id=i) 
            for i in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        hidden_states = self.embeddings(input_ids) + self.pos_embeddings(positions)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_attentions:
                hidden_states, attn_weights = layer(hidden_states, attention_mask, output_attentions)
                all_attentions = all_attentions + (attn_weights,)
            else:
                hidden_states = layer(hidden_states, attention_mask, output_attentions)
        
        if output_attentions:
            return hidden_states, all_attentions
        return hidden_states


class StandardDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(
            self.embed_dim, 
            config.num_attention_heads, 
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Cross attention
        self.encoder_attn = nn.MultiheadAttention(
            self.embed_dim, 
            config.num_attention_heads, 
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Feed forward
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = nn.GELU()
        self.dropout = config.hidden_dropout_prob

    def forward(self, hidden_states, encoder_hidden_states, 
                tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self attention
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross attention
        residual = hidden_states
        hidden_states, _ = self.encoder_attn(
            hidden_states, encoder_hidden_states, encoder_hidden_states,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Feed forward
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class StandardDecoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model
        
        self.embeddings = nn.Embedding(vocab.vocab_size, self.embed_dim, padding_idx=vocab.pad_idx)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([
            StandardDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.output_projection = nn.Linear(self.embed_dim, vocab.vocab_size)

    def forward(self, input_ids, encoder_hidden_states, tgt_mask=None, src_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        hidden_states = self.embeddings(input_ids) + self.pos_embeddings(positions)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create attention masks
        tgt_key_padding_mask = (input_ids == self.config.pad_idx) if hasattr(self.config, 'pad_idx') else None
        memory_key_padding_mask = None  # Encoder typically handles this
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        logits = self.output_projection(hidden_states)
        return logits


@META_ARCHITECTURE.register()
class LongformerEncoderDecoderModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        
        # Store vocab indices
        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_bos_idx = vocab.bos_idx
        self.trg_eos_idx = vocab.eos_idx
        
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        
        # Initialize encoder and decoder
        self.encoder = LongformerEncoder(config, vocab)
        self.decoder = StandardDecoder(config, vocab)
        
        self.d_model = config.d_model
        self.device = config.device
        self.config = config
        self.vocab = vocab
        
        # Store pad idx in config for decoder
        self.config.pad_idx = vocab.pad_idx

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, src, trg):
        """
        Fixed forward method that always returns (outputs, loss) like LSTM model
        """
        src = self._pad_to_window_size(src)
        
        # Create masks
        src_attention_mask = self.make_longformer_attention_mask(src)
        
        # For training with labels
        if trg is not None:
            # Decoder input (shifted right, starting with BOS) 
            decoder_input = trg[:, :-1]  # Remove last token for input
            # Target tokens (shifted left, ending with EOS)
            decoder_target = trg[:, 1:]  # Remove first token for target
            
            # Create target mask for decoder input
            trg_mask = self.make_trg_mask(decoder_input)
            
            # Encode
            encoder_outputs = self.encoder(src, attention_mask=src_attention_mask)
            
            # Decode
            decoder_outputs = self.decoder(decoder_input, encoder_outputs, tgt_mask=trg_mask)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                decoder_target.contiguous().view(-1),
                ignore_index=self.trg_pad_idx
            )
            
            return decoder_outputs, loss
        else:
            # Inference mode - should not happen during training
            # But return dummy loss for consistency
            encoder_outputs = self.encoder(src, attention_mask=src_attention_mask)
            
            # Create a dummy decoder output and loss
            batch_size, seq_len = src.shape
            vocab_size = self.decoder.output_projection.out_features
            dummy_outputs = torch.zeros(batch_size, seq_len, vocab_size, device=src.device)
            dummy_loss = torch.tensor(0.0, device=src.device, requires_grad=True)
            
            return dummy_outputs, dummy_loss

    def _pad_to_window_size(self, input_ids):
        config = self.config
        
        w = config.attention_window[0] * 2
        seq_len = input_ids.size(1)
        padding_len = (w - seq_len % w) % w 
        
        if padding_len > 0:
            # Use self.src_pad_idx instead of self.vocab.pad_idx
            input_ids = F.pad(input_ids, (0, padding_len), value=self.src_pad_idx)
        
        return input_ids
        

    def make_longformer_attention_mask(self, src, use_global_for_bos=True):
        """
        Creates attention mask for Longformer:
        - Negative values (-1): no attention (padding)
        - Zero (0): local windowed attention
        - Positive values (>0): global attention
        """
        attention_mask = torch.zeros_like(src, dtype=torch.long)
        
        # Mark padding positions with -1
        attention_mask = attention_mask.masked_fill(src == self.src_pad_idx, -1)
        
        # Optionally mark BOS tokens with global attention
        if use_global_for_bos and hasattr(self, 'trg_bos_idx'):
            attention_mask = attention_mask.masked_fill(src == self.trg_bos_idx, 2)
        
        return attention_mask.unsqueeze(1).unsqueeze(1)
    
    

    def make_trg_mask(self, trg):
        trg_len = trg.shape[1]
        # Create causal mask
        trg_mask = torch.tril(torch.ones(trg_len, trg_len, device=trg.device))
        # Convert to boolean and invert (True = masked positions)
        trg_mask = (trg_mask == 0)
        return trg_mask

    def predict(self, src: torch.Tensor) -> torch.Tensor:
        config = self.config
        
        # Pad source to window size
        src = self._pad_to_window_size(src)
        
        # Create source attention mask
        src_attention_mask = self.make_longformer_attention_mask(src)
        
        # Encode
        encoder_outputs = self.encoder(src, attention_mask=src_attention_mask)
        
        # Initialize decoder input with BOS token
        batch_size = src.size(0)
        decoder_input = torch.full((batch_size, 1), self.trg_bos_idx, 
                                dtype=torch.long, device=src.device)
        
        outputs = []
        
        # Generate tokens one by one
        for step in range(self.MAX_LENGTH):
            # Create target mask
            trg_mask = self.make_trg_mask(decoder_input)
            
            # Get decoder output
            decoder_output = self.decoder(decoder_input, encoder_outputs, tgt_mask=trg_mask)
            
            # Get next token (argmax of last position)
            next_token = decoder_output[:, -1:, :].argmax(dim=-1)
            
            # Add to outputs
            outputs.append(next_token)
            
            # Update decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Check for EOS token (assuming batch_size=1 for simplicity)
            if next_token.item() == self.trg_eos_idx:
                break
        
        # Concatenate all outputs
        if outputs:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = torch.empty(batch_size, 0, dtype=torch.long, device=src.device)
        
        return outputs
    
#  python3 main.py --config-file configs/longformer_Wikilingual.yaml

