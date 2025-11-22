from typing import List, Tuple, Optional, Dict
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import Counter, defaultdict
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

INI = 1e-2


def len_mask(lens, device):
    """Return boolean mask [B, max_len] where True indicates valid positions."""
    if isinstance(lens, torch.Tensor):
        lens = lens.tolist()
    if not isinstance(lens, (list, tuple)) or len(lens) == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=device)
    max_len = max(lens)
    batch_size = len(lens)
    arange = torch.arange(max_len, device=device).expand(batch_size, max_len)
    lens_tensor = torch.tensor(lens, device=device).unsqueeze(1)
    return arange < lens_tensor


def sequence_mean(sequence: torch.Tensor, seq_lens: Optional[List[int]] = None, dim: int = 1):
    """Mean pooling over valid positions using seq_lens."""
    if seq_lens is None:
        return sequence.mean(dim=dim)
    mask = len_mask(seq_lens, sequence.device).unsqueeze(-1).float()
    summed = (sequence * mask).sum(dim=dim)
    lens_tensor = torch.tensor(seq_lens, device=sequence.device).unsqueeze(-1).float().clamp(min=1.0)
    return summed / lens_tensor


def prob_normalize(score: torch.Tensor, mask: torch.Tensor):
    """Masked softmax with safety when entire row masked."""
    score = score.masked_fill(~mask, -1e18)
    all_masked = (~mask).all(dim=-1)
    if all_masked.any():
        score = score.masked_fill(all_masked.unsqueeze(-1), 0.0)
    return F.softmax(score, dim=-1)


def step_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                  mem_mask: Optional[torch.Tensor] = None):
    """
    Compute single-step attention.
    query: [B, Dq]
    key: [B, L, Dk]
    value: [B, L, Dv]
    mem_mask: [B, 1, L] or [B, L]
    Returns: (context [B, Dv], norm_score [B, L])
    """
    score = torch.bmm(query.unsqueeze(1), key.transpose(1, 2))  # [B, 1, L]
    if mem_mask is not None:
        m = mem_mask.unsqueeze(1) if mem_mask.dim() == 2 else mem_mask
        norm = prob_normalize(score, m)
    else:
        norm = F.softmax(score, dim=-1)
    ctx = torch.bmm(norm, value).squeeze(1)  # [B, Dv]
    return ctx, norm.squeeze(1)


def reorder_sequence(sequence_emb: torch.Tensor, order: List[int], batch_first: bool = True):
    """Reorder sequence by given indices."""
    dim = 0 if batch_first else 1
    order_t = torch.LongTensor(order).to(sequence_emb.device)
    return sequence_emb.index_select(dim, order_t)


def reorder_lstm_states(lstm_states: Tuple[torch.Tensor, torch.Tensor], order: List[int]):
    """Reorder LSTM hidden states."""
    order_t = torch.LongTensor(order).to(lstm_states[0].device)
    h = lstm_states[0].index_select(1, order_t)
    c = lstm_states[1].index_select(1, order_t)
    return (h, c)


def lstm_encoder(sequence: torch.Tensor, lstm: nn.LSTM, seq_lens: Optional[List[int]] = None,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                embedding: Optional[nn.Embedding] = None):
    """
    Encode sequence with LSTM, handling variable lengths with packing.
    sequence: [B, L] token ids or [B, L, D_emb] embeddings
    Returns: (lstm_out [B, L, D], (h, c))
    """
    if embedding is not None:
        sequence = embedding(sequence)
    
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1)
    
    if seq_lens is not None:
        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.cpu().tolist()
        
        # Sort by length
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
        seq_lens_sorted = [seq_lens[i] for i in sort_ind]
        sequence = reorder_sequence(sequence, sort_ind, batch_first=lstm.batch_first)
        
        # Pack and encode
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, seq_lens_sorted, batch_first=lstm.batch_first
        )
        packed_out, final_states = lstm(packed, init_states)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=lstm.batch_first)
        
        # Restore original order
        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(sort_ind))]
        out = reorder_sequence(out, reorder_ind, batch_first=lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        out, final_states = lstm(sequence, init_states)
    
    return out, final_states


def _has_repeat_tri(grams: List[int]):
    """Check for repeated trigrams."""
    if len(grams) < 3:
        return False
    tri = [tuple(grams[i:i+3]) for i in range(len(grams) - 2)]
    cnt = Counter(tri)
    return any(c > 1 for c in cnt.values())


class ConvEncoder(nn.Module):
    """Multi-kernel 1D convolutional encoder."""
    
    def __init__(self, emb_dim: int, n_hidden: int, kernel_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        self._convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=n_hidden, 
                     kernel_size=k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self._dropout = dropout
        self._n_kernels = len(kernel_sizes)
        self._n_hidden = n_hidden

    def forward(self, emb_input: torch.Tensor):
        """
        Args:
            emb_input: [B, L, D_emb]
        Returns:
            [B, L, n_hidden * n_kernels]
        """
        x = emb_input.transpose(1, 2)  # [B, D, L]
        x = F.dropout(x, p=self._dropout, training=self.training)
        outs = [F.relu(conv(x)) for conv in self._convs]
        
        # Align lengths
        min_len = min(o.size(2) for o in outs)
        outs = [o[:, :, :min_len] for o in outs]
        
        cat = torch.cat(outs, dim=1)  # [B, n_hidden * n_kernels, L]
        return cat.transpose(1, 2)  # [B, L, n_hidden * n_kernels]


class MultiLayerLSTMCells(nn.Module):
    """Stacked LSTM cells for step-by-step decoding."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self._dropout = dropout
        
        cells = []
        for i in range(num_layers):
            insz = input_size if i == 0 else hidden_size
            cells.append(nn.LSTMCell(insz, hidden_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, input_: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            input_: [B, input_size]
            states: (h_prev, c_prev) each [num_layers, B, hidden]
        Returns:
            layer_output: [B, hidden]
            (new_h, new_c): each [num_layers, B, hidden]
        """
        h_prev, c_prev = states
        new_h, new_c = [], []
        
        for i, cell in enumerate(self.cells):
            h_i = h_prev[i]
            c_i = c_prev[i]
            
            layer_input = input_ if i == 0 else layer_output
            h_i, c_i = cell(layer_input, (h_i, c_i))
            layer_output = h_i
            
            if self._dropout > 0 and i < self.num_layers - 1 and self.training:
                layer_output = F.dropout(layer_output, p=self._dropout)
            
            new_h.append(h_i)
            new_c.append(c_i)
        
        new_h = torch.stack(new_h, dim=0)
        new_c = torch.stack(new_c, dim=0)
        return layer_output, (new_h, new_c)



class _CopyLinear(nn.Module):
    """Compute copy gate: v_c^T c + v_s^T s + v_i^T x + b"""
    
    def __init__(self, context_dim: int, state_dim: int, input_dim: int, bias: bool = True):
        super().__init__()
        self._v_c = nn.Parameter(torch.empty(context_dim))
        self._v_s = nn.Parameter(torch.empty(state_dim))
        self._v_i = nn.Parameter(torch.empty(input_dim))
        
        init = torch.nn.init.uniform_
        init(self._v_c, -INI, INI)
        init(self._v_s, -INI, INI)
        init(self._v_i, -INI, INI)
        
        self._b = nn.Parameter(torch.zeros(1)) if bias else None

    def forward(self, context: torch.Tensor, state: torch.Tensor, input_: torch.Tensor):
        """
        Args:
            context: [B, context_dim]
            state: [B, state_dim]
            input_: [B, input_dim]
        Returns:
            [B, 1] copy gate logits
        """
        out = (context @ self._v_c.unsqueeze(-1) +
               state @ self._v_s.unsqueeze(-1) +
               input_ @ self._v_i.unsqueeze(-1))
        if self._b is not None:
            out = out + self._b
        return out


class CopyLSTMDecoder(nn.Module):
    """LSTM decoder with copy mechanism - GREEDY SEARCH version."""
    
    def __init__(self, copy_linear: _CopyLinear, embedding: nn.Embedding,
                lstm_cells: MultiLayerLSTMCells, attn_w: torch.Tensor, projection: nn.Module):
        super().__init__()
        self._copy = copy_linear
        self._embedding = embedding
        self._lstm = lstm_cells
        self._attn_w = attn_w
        self._projection = projection

    def _step(self, tok: torch.Tensor, states, attention):
        """
        Single decoding step.
        Args:
            tok: [B, 1] token indices
            states: ((h, c), prev_out)
            attention: (enc_mem, enc_proj, mask, extend_art, extend_vsize)
        Returns:
            log_prob: [B, extend_vsize]
            new_states: ((h, c), dec_out)
            score: [B, L] attention scores
        """
        (prev_h, prev_c), prev_out = states
        
        # Embedding
        emb = self._embedding(tok).squeeze(1)  # [B, emb]
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        
        if prev_out.dim() == 3:
            prev_out = prev_out.squeeze(1)
        
        # LSTM input: concat embedding and previous output
        lstm_in = torch.cat([emb, prev_out], dim=-1)
        
        # LSTM step
        lstm_out, (new_h, new_c) = self._lstm(lstm_in, (prev_h, prev_c))
        
        # Attention
        enc_mem, enc_proj, mask, extend_art, extend_vsize = attention
        query = lstm_out @ self._attn_w
        ctx, score = step_attention(query, enc_proj, enc_mem, mask)
        
        # Decoder output projection
        dec_out = self._projection(torch.cat([lstm_out, ctx], dim=-1))
        
        # Generation distribution
        logits = dec_out @ self._embedding.weight.T  # [B, vocab_size]
        if extend_vsize > logits.size(1):
            pad = torch.full((logits.size(0), extend_vsize - logits.size(1)), 
                           -1e8, device=logits.device)
            logits = torch.cat([logits, pad], dim=1)
        gen_prob = F.softmax(logits, dim=-1)
        
        # Copy gate
        copy_gate = torch.sigmoid(self._copy(ctx, lstm_out, emb))
        if copy_gate.dim() == 1:
            copy_gate = copy_gate.unsqueeze(-1)
        
        # Copy distribution
        copy_vals = score * copy_gate
        add_tensor = torch.zeros_like(gen_prob)
        
        if extend_art.dim() == 1:
            extend_art = extend_art.unsqueeze(0).expand(add_tensor.size(0), -1)
        
        # Clamp indices to valid range
        if extend_art.max().item() >= extend_vsize:
            extend_art = extend_art.clamp(max=extend_vsize - 1)
        
        add_tensor.scatter_add_(1, extend_art.long(), copy_vals)
        
        # Final probability: generation + copy
        final_prob = (1 - copy_gate) * gen_prob + add_tensor
        log_prob = torch.log(final_prob + 1e-12)
        
        new_states = ((new_h, new_c), dec_out)
        return log_prob, new_states, score

    def forward(self, attention, init_states, abstract):
        """
        Teacher forcing decoding for training.
        Args:
            attention: (enc_mem, enc_proj, mask, extend_art, extend_vsize)
            init_states: ((h, c), prev_out)
            abstract: [B, T] target sequence
        Returns:
            log_probs: [B, T, extend_vsize]
        """
        B, T = abstract.size()
        states = init_states
        outputs = []
        
        for t in range(T):
            tok = abstract[:, t:t+1]
            logp, states, _ = self._step(tok, states, attention)
            outputs.append(logp)
        
        return torch.stack(outputs, dim=1)

    def greedy_decode(self, attention, init_states, go: int, eos: int, 
                     max_len: int, min_len: int = 1):
        """
        Greedy decoding for inference.
        
        Args:
            attention: (enc_mem, enc_proj, mask, extend_art, extend_vsize)
            init_states: ((h, c), prev_out)
            go: int BOS token
            eos: int EOS token
            max_len: int maximum generation length
            min_len: int minimum generation length
        
        Returns:
            generated: [B, generated_len] generated token sequence
            log_probs: [B, generated_len] log probabilities
        """
        (h, c), prev_out = init_states
        B = h.size(1)
        device = h.device
        
        # Initialize with BOS token
        current_token = torch.full((B, 1), go, dtype=torch.long, device=device)
        
        generated_tokens = []
        generated_logprobs = []
        states = init_states
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            # Decode step
            log_prob, states, attn_score = self._step(current_token, states, attention)
            
            # Force not to stop before min_len
            if t < min_len:
                log_prob[:, eos] = -1e20
            
            # Greedy selection
            selected_logprob, selected_token = log_prob.max(dim=-1)
            
            # Store results
            generated_tokens.append(selected_token)
            generated_logprobs.append(selected_logprob)
            
            # Update finished status
            finished = finished | (selected_token == eos)
            
            # Early stopping if all sequences finished
            if finished.all():
                break
            
            # Prepare next input
            current_token = selected_token.unsqueeze(1)
        
        # Stack results
        generated = torch.stack(generated_tokens, dim=1)  # [B, T]
        log_probs = torch.stack(generated_logprobs, dim=1)  # [B, T]
        
        return generated, log_probs



class LSTMPointerNet(nn.Module):
    """Pointer network decoder with entity attention."""
    
    def __init__(self, input_dim: int, n_hidden: int, n_layer: int, 
                dropout: float, n_hop: int, side_dim: int):
        super().__init__()
        self._lstm = nn.LSTM(input_dim, n_hidden, num_layers=n_layer, 
                            batch_first=True, dropout=dropout if n_layer > 1 else 0.0)
        
        # Attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        
        # Entity attention parameters
        self.side_wm = nn.Parameter(torch.Tensor(side_dim, n_hidden))
        self.side_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.side_v = nn.Parameter(torch.Tensor(n_hidden))
        
        self._attn_ws = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        
        # Initialize
        for p in [self._attn_wm, self._attn_wq, self.side_wm, self.side_wq, self._attn_ws]:
            nn.init.xavier_normal_(p)
        for p in [self._attn_v, self.side_v]:
            nn.init.uniform_(p, -INI, INI)

    def forward(self, sent_mem: torch.Tensor, entity_mem: torch.Tensor, 
               ptr_in: torch.Tensor, sent_nums: List[int], entity_nums: List[int]):
        """
        Args:
            sent_mem: [B, Ns, D_sent]
            entity_mem: [B, Ne, D_entity]
            ptr_in: [B, T, D_sent]
            sent_nums: List[int]
            entity_nums: List[int]
        Returns:
            score: [B, T, Ns]
        """
        query, _ = self._lstm(ptr_in)  # [B, T, n_hidden]
        
        # Entity attention
        side_feat = entity_mem @ self.side_wm
        entity_ctx = self._attention(side_feat, query, self.side_v, self.side_wq, entity_nums)
        
        # Sentence attention (with entity context)
        sent_feat = sent_mem @ self._attn_wm
        score = self._attn_with_side(sent_feat, query, entity_ctx, 
                                     self._attn_v, self._attn_wq, self._attn_ws, sent_nums)
        return score

    @staticmethod
    def _attention(mem, query, v, w, sizes):
        """Standard attention."""
        score = torch.tanh(mem.unsqueeze(1) + query.matmul(w).unsqueeze(2))
        score = score.matmul(v.unsqueeze(-1)).squeeze(-1)
        mask = len_mask(sizes, score.device).unsqueeze(1)
        score = score.masked_fill(~mask, -1e9)
        attn = F.softmax(score, dim=-1)
        return attn.matmul(mem)

    @staticmethod
    def _attn_with_side(mem, query, ctx, v, wq, ws, sizes):
        """Attention with side information (entity context)."""
        s = query.matmul(wq).unsqueeze(2)
        e = ctx.matmul(ws).unsqueeze(2)
        score = torch.tanh(mem.unsqueeze(1) + s + e).matmul(v.unsqueeze(-1)).squeeze(-1)
        mask = len_mask(sizes, score.device).unsqueeze(1)
        score = score.masked_fill(~mask, -1e9)
        return score


class EntityAwareExtractor(nn.Module):
    """Entity-aware content selector."""
    
    def __init__(self, vocab_size: int, emb_dim: int, conv_hidden: int, 
                lstm_hidden: int, lstm_layer: int, bidirectional: bool, 
                n_hop: int, dropout: float):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Sentence encoder
        self.sent_conv_encoder = ConvEncoder(
            emb_dim=emb_dim, n_hidden=conv_hidden, 
            kernel_sizes=[3, 4, 5], dropout=dropout
        )
        self.sent_lstm_encoder = nn.LSTM(
            input_size=3 * conv_hidden, hidden_size=lstm_hidden, 
            num_layers=lstm_layer, bidirectional=bidirectional, 
            batch_first=True, dropout=dropout if lstm_layer > 1 else 0.0
        )
        
        # Entity encoder
        self.entity_conv_encoder = ConvEncoder(
            emb_dim=emb_dim, n_hidden=conv_hidden, 
            kernel_sizes=[2, 3, 4], dropout=dropout
        )
        
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        entity_dim = 3 * conv_hidden
        
        # Project entity to match sentence dimension
        self.entity_proj = nn.Linear(entity_dim, enc_out_dim)
        
        # Pointer network
        self.pointer_decoder = LSTMPointerNet(
            input_dim=enc_out_dim, n_hidden=lstm_hidden, 
            n_layer=lstm_layer, dropout=dropout, 
            n_hop=n_hop, side_dim=enc_out_dim
        )

    def _encode_sentences(self, article_sents: torch.Tensor, sent_nums: List[int]):
        """Encode sentences with CNN + BiLSTM."""
        B, max_sents, max_words = article_sents.size()
        sents_emb = self._embedding(article_sents)
        
        # CNN encoding
        sents_flat = sents_emb.view(B * max_sents, max_words, -1)
        conv_sents = self.sent_conv_encoder(sents_flat)
        sent_vecs = conv_sents.max(dim=1).values  # Max pooling
        sent_vecs = sent_vecs.view(B, max_sents, -1)
        
        # BiLSTM encoding
        if isinstance(sent_nums, torch.Tensor):
            sent_nums = sent_nums.cpu().tolist()
        
        packed = nn.utils.rnn.pack_padded_sequence(
            sent_vecs, sent_nums, batch_first=True, enforce_sorted=False
        )
        sent_mem, _ = self.sent_lstm_encoder(packed)
        sent_mem, _ = nn.utils.rnn.pad_packed_sequence(sent_mem, batch_first=True)
        
        return sent_mem  # [B, Ns, enc_out_dim]

    def _encode_entities(self, clusters: torch.Tensor, cluster_nums: List[int]):
        """Encode entity clusters with CNN."""
        B, max_clusters, max_mentions = clusters.size()
        ent_emb = self._embedding(clusters)
        
        # CNN encoding
        ent_flat = ent_emb.view(B * max_clusters, max_mentions, -1)
        conv_ents = self.entity_conv_encoder(ent_flat)
        entity_vecs = conv_ents.max(dim=1).values  # Max pooling
        entity_mem = entity_vecs.view(B, max_clusters, -1)
        
        return entity_mem

    def forward(self, article_sents: torch.Tensor, sent_nums: torch.Tensor,
                clusters: torch.Tensor, cluster_nums: torch.Tensor,
                target_indices: Optional[torch.Tensor] = None,
                max_extract: int = 5):
        """
        Forward pass for training or inference.
        
        Args:
            article_sents: [B, Ns, Nw] sentence tokens
            sent_nums: [B] number of sentences per sample
            clusters: [B, Ne, Nm] entity cluster tokens
            cluster_nums: [B] number of clusters per sample
            target_indices: [B, T] target extraction indices (for training)
            max_extract: maximum sentences to extract (for inference)
        
        Returns:
            Training: scores [B, T, Ns]
            Inference: List[List[int]] extracted indices
        """
        # Encode
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        entity_mem = self._encode_entities(clusters, cluster_nums)
        
        B, Ns, D_sent = sent_mem.size()
        
        # Project entity to match sentence dimension
        entity_proj = self.entity_proj(entity_mem)
        
        sent_nums_list = sent_nums.cpu().tolist() if isinstance(sent_nums, torch.Tensor) else sent_nums
        cluster_nums_list = cluster_nums.cpu().tolist() if isinstance(cluster_nums, torch.Tensor) else cluster_nums
        
        # Initialize pointer input
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + entity_proj.mean(dim=1, keepdim=True)) / 2
        
        # Training mode: use target indices
        if target_indices is not None:
            T = target_indices.size(1)
            scores_all = []
            
            for t in range(T):
                score = self.pointer_decoder(
                    sent_mem, entity_proj, ptr_in, sent_nums_list, cluster_nums_list
                )
                scores_all.append(score.squeeze(1))
                
                # Update pointer input with selected sentence
                idx = target_indices[:, t].clamp(0, Ns - 1)
                ptr_in = torch.gather(
                    sent_mem, dim=1, 
                    index=idx.unsqueeze(-1).unsqueeze(-1).expand(B, 1, D_sent)
                )
            
            scores = torch.stack(scores_all, dim=1)  # [B, T, Ns]
            return scores
        
        # Inference mode: autoregressive extraction (GREEDY)
        else:
            extracted = [[] for _ in range(B)]
            
            for step in range(max_extract):
                score = self.pointer_decoder(
                    sent_mem, entity_proj, ptr_in, sent_nums_list, cluster_nums_list
                )
                score = score.squeeze(1)  # [B, Ns]
                
                # Mask invalid positions
                mask = len_mask(sent_nums_list, score.device)
                score = score.masked_fill(~mask, -1e9)
                
                # Greedy selection
                probs = F.softmax(score, dim=-1)
                selected = torch.argmax(probs, dim=-1)
                
                # Add to extracted (avoid duplicates)
                for b in range(B):
                    idx = selected[b].item()
                    if idx not in extracted[b]:
                        extracted[b].append(idx)
                
                # Update pointer input
                ptr_in = torch.gather(
                    sent_mem, dim=1, 
                    index=selected.unsqueeze(-1).unsqueeze(-1).expand(B, 1, D_sent)
                )
            
            return extracted

    @torch.no_grad()
    def extract(self, article_sents: torch.Tensor, sent_nums: torch.Tensor, 
               clusters: torch.Tensor, cluster_nums: torch.Tensor, k: int = 4):
        """
        Extract top-k sentences (non-autoregressive).
        
        Returns:
            List[List[int]] - top-k sentence indices per sample
        """
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        entity_mem = self._encode_entities(clusters, cluster_nums)
        entity_proj = self.entity_proj(entity_mem)
        
        B = article_sents.size(0)
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + entity_proj.mean(dim=1, keepdim=True)) / 2
        
        sent_nums_list = sent_nums.cpu().tolist() if isinstance(sent_nums, torch.Tensor) else sent_nums
        cluster_nums_list = cluster_nums.cpu().tolist() if isinstance(cluster_nums, torch.Tensor) else cluster_nums
        
        scores = self.pointer_decoder(
            sent_mem=sent_mem, entity_mem=entity_proj, ptr_in=ptr_in,
            sent_nums=sent_nums_list, entity_nums=cluster_nums_list
        )
        scores = scores.squeeze(1)
        
        # Mask and select top-k
        mask = len_mask(sent_nums_list, scores.device)
        scores = scores.masked_fill(~mask, -1e9)
        probs = F.softmax(scores, dim=-1)
        _, topk = torch.topk(probs, k, dim=-1)
        
        return [idx.cpu().tolist() for idx in topk]

    @torch.no_grad()
    def extract_greedy(self, article_sents: torch.Tensor, sent_nums: torch.Tensor,
                      clusters: torch.Tensor, cluster_nums: torch.Tensor, 
                      k: int = 4):
        """
        Extract sentences greedily with autoregressive masking.
        
        Returns:
            List[List[int]] - extracted sentence indices
        """
        sent_mem = self._encode_sentences(article_sents, sent_nums)
        entity_mem = self._encode_entities(clusters, cluster_nums)
        entity_proj = self.entity_proj(entity_mem)
        
        B = article_sents.size(0)
        sent_nums_list = sent_nums.cpu().tolist() if isinstance(sent_nums, torch.Tensor) else sent_nums
        cluster_nums_list = cluster_nums.cpu().tolist() if isinstance(cluster_nums, torch.Tensor) else cluster_nums
        
        extracted = [[] for _ in range(B)]
        ptr_in = (sent_mem.mean(dim=1, keepdim=True) + entity_proj.mean(dim=1, keepdim=True)) / 2
        
        for step in range(k):
            scores = self.pointer_decoder(
                sent_mem=sent_mem, entity_mem=entity_proj, ptr_in=ptr_in,
                sent_nums=sent_nums_list, entity_nums=cluster_nums_list
            )
            scores = scores.squeeze(1)
            
            # Mask already extracted and invalid positions
            for b in range(B):
                for idx in extracted[b]:
                    scores[b, idx] = -1e9
                for idx in range(sent_nums_list[b], scores.size(1)):
                    scores[b, idx] = -1e9
            
            probs = F.softmax(scores, dim=-1)
            selected = torch.argmax(probs, dim=-1)
            
            # Add to extracted
            for b in range(B):
                idx = selected[b].item()
                if idx < sent_nums_list[b] and idx not in extracted[b]:
                    extracted[b].append(idx)
            
            # Update pointer input
            ptr_in = torch.gather(
                sent_mem, dim=1,
                index=selected.unsqueeze(-1).unsqueeze(-1).expand(B, 1, sent_mem.size(-1))
            )
            
            # Check if all samples have k sentences
            done = [len(ext) >= min(k, sent_nums_list[b]) for b, ext in enumerate(extracted)]
            if all(done):
                break
        
        return extracted



class AbstractGenerator(nn.Module):
    """Abstractive summary generator with attention, copy mechanism, and GREEDY decoding."""
    
    def __init__(self, vocab_size: int, emb_dim: int, n_hidden: int, 
                bidirectional: bool, n_layer: int, dropout: float = 0.1):
        super().__init__()
        self._vocab_size = vocab_size
        self._n_hidden = n_hidden
        self._n_layer = n_layer
        self._bidirectional = bidirectional
        
        # Embedding (shared between encoder and decoder)
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Encoder: BiLSTM
        self._enc_lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=n_hidden, 
            num_layers=n_layer, bidirectional=bidirectional,
            batch_first=True, dropout=dropout if n_layer > 1 else 0.0
        )
        
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        
        # Decoder: LSTM cells
        self._dec_lstm_cells = MultiLayerLSTMCells(
            input_size=emb_dim * 2,  # embedding + prev attention output
            hidden_size=enc_out_dim,
            num_layers=n_layer,
            dropout=dropout
        )
        
        # Initial state projection
        self._dec_h = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, enc_out_dim, bias=False)
        
        # Attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, enc_out_dim))
        self._attn_wq = nn.Parameter(torch.Tensor(enc_out_dim, enc_out_dim))
        nn.init.xavier_normal_(self._attn_wm)
        nn.init.xavier_normal_(self._attn_wq)
        
        # Output projection
        self._projection = nn.Sequential(
            nn.Linear(2 * enc_out_dim, enc_out_dim),
            nn.Tanh(),
            nn.Linear(enc_out_dim, emb_dim, bias=False)
        )
        
        # Copy mechanism
        self._copy = _CopyLinear(
            context_dim=enc_out_dim,
            state_dim=enc_out_dim,
            input_dim=emb_dim
        )
        
        # Decoder
        self._decoder = CopyLSTMDecoder(
            copy_linear=self._copy,
            embedding=self._embedding,
            lstm_cells=self._dec_lstm_cells,
            attn_w=self._attn_wq,
            projection=self._projection
        )

    def encode(self, article: torch.Tensor, art_lens: Optional[List[int]] = None):
        """
        Encode article with BiLSTM.
        
        Args:
            article: [B, L] token ids
            art_lens: List[int] actual lengths
        
        Returns:
            (enc_art, enc_proj): encoded representations
            init_states: initial decoder states
        """
        enc_art, (h, c) = lstm_encoder(
            sequence=article, lstm=self._enc_lstm, 
            seq_lens=art_lens, embedding=self._embedding
        )
        
        # Handle bidirectional
        if self._bidirectional:
            h = h.view(self._n_layer, 2, h.size(1), h.size(2))
            c = c.view(self._n_layer, 2, c.size(1), c.size(2))
            h = torch.cat([h[:, 0], h[:, 1]], dim=2)
            c = torch.cat([c[:, 0], c[:, 1]], dim=2)
        
        # Initialize decoder states
        init_h = torch.stack([self._dec_h(s) for s in h], dim=0)
        init_c = torch.stack([self._dec_c(s) for s in c], dim=0)
        
        # Project encoder outputs for attention
        enc_proj = torch.matmul(enc_art, self._attn_wm)
        
        # Initialize previous attention output
        mean_ctx = sequence_mean(enc_proj, art_lens, dim=1)
        prev_attn_out = self._projection(torch.cat([init_h[-1], mean_ctx], dim=1))
        
        init_states = ((init_h, init_c), prev_attn_out)
        
        return (enc_art, enc_proj), init_states

    def forward(self, article: torch.Tensor, art_lens: List[int], 
               abstract: torch.Tensor, extend_art: torch.Tensor, extend_vsize: List[int]):
        """
        Forward pass for training (teacher forcing).
        
        Args:
            article: [B, L] source tokens
            art_lens: List[int] source lengths
            abstract: [B, T] target tokens
            extend_art: [B, L] extended vocab indices
            extend_vsize: int extended vocab size
        
        Returns:
            logits: [B, T, extend_vsize] log probabilities
        """
        # Encode
        (enc_art, enc_proj), init_states = self.encode(article, art_lens)
        
        # Prepare attention input
        mask = len_mask(art_lens, enc_art.device).unsqueeze(-2)
        decoder_input = (enc_art, enc_proj, mask, extend_art.to(enc_art.device), extend_vsize)
        
        # Decode with teacher forcing
        logits = self._decoder(decoder_input, init_states, abstract)
        
        return logits

    @torch.no_grad()
    def greedy_generate(self, article: torch.Tensor, art_lens: List[int],
                       extend_art: torch.Tensor, extend_vsize: int,
                       go: int, eos: int, max_len: int, min_len: int = 1):
        """
        Greedy generation for inference.
        
        Args:
            article: [B, L] source tokens
            art_lens: List[int] source lengths
            extend_art: [B, L] extended vocab indices
            extend_vsize: int extended vocab size
            go: int BOS token
            eos: int EOS token
            max_len: int max generation length
            min_len: int minimum generation length
        
        Returns:
            List[List[int]] - generated token sequences
        """
        device = article.device
        batch_size = article.size(0)
        
        # Encode
        (enc_art, enc_proj), init_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, device).unsqueeze(-2)
        attention_input = (enc_art, enc_proj, mask, extend_art.to(device), extend_vsize)
        
        # Greedy decode
        generated, log_probs = self._decoder.greedy_decode(
            attention_input, init_states, go, eos, max_len, min_len
        )
        
        # Convert to list of sequences
        outputs = []
        for i in range(batch_size):
            seq = generated[i].cpu().tolist()
            # Remove tokens after EOS
            if eos in seq:
                eos_idx = seq.index(eos)
                seq = seq[:eos_idx]
            outputs.append(seq)
        
        return outputs


@META_ARCHITECTURE.register()
class SENECAModel(nn.Module):
    """
    Complete SENECA model for abstractive summarization with GREEDY SEARCH.
    Combines entity-aware content selection and abstract generation.
    """
    
    def __init__(self, cfg, vocab):
        """
        Args:
            cfg: configuration object with SENECA.EXTRACTOR and SENECA.GENERATOR
            vocab: vocabulary object
        """
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.vocab_size = len(vocab) if hasattr(vocab, '__len__') else vocab
        self.d_model = getattr(cfg, "d_model", None)
        
        # Get configurations
        ext_cfg = cfg.SENECA.EXTRACTOR
        gen_cfg = cfg.SENECA.GENERATOR
        
        # Entity-aware extractor (content selector)
        self.extractor = EntityAwareExtractor(
            vocab_size=self.vocab_size,
            emb_dim=ext_cfg.EMB_DIM,
            conv_hidden=ext_cfg.CONV_HIDDEN,
            lstm_hidden=ext_cfg.LSTM_HIDDEN,
            lstm_layer=ext_cfg.LSTM_LAYER,
            bidirectional=ext_cfg.BIDIRECTIONAL,
            n_hop=ext_cfg.N_HOP,
            dropout=ext_cfg.DROPOUT
        )
        
        # Abstract generator
        self.generator = AbstractGenerator(
            vocab_size=self.vocab_size,
            emb_dim=gen_cfg.EMB_DIM,
            n_hidden=gen_cfg.N_HIDDEN,
            bidirectional=gen_cfg.BIDIRECTIONAL,
            n_layer=gen_cfg.N_LAYER,
            dropout=gen_cfg.DROPOUT
        )
        
        # Loss functions
        self.extraction_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=0)

    def _build_extended_for_batch(self, articles: torch.Tensor):
        """
        Build extended vocab mapping for flattened articles.
        articles: [B, L]             (flattened sentences, padded)
        Returns:
            extend_art:  [B, L']     extended indices, padded
            extend_vsize: int        max extended size across batch
        """
        B, L = articles.size()
        device = articles.device
        vocab_size = self.vocab_size

        extend_list = []
        extend_vsizes = []

        for b in range(B):
            # non-pad tokens only
            nonpad = articles[b][articles[b] != 0].tolist()

            oov2idx = {}
            extended = []

            for tok in nonpad:
                if tok < vocab_size:
                    extended.append(tok)
                else:
                    if tok not in oov2idx:
                        oov2idx[tok] = len(oov2idx)
                    extended.append(vocab_size + oov2idx[tok])

            # Store results
            extend_list.append(torch.tensor(extended, device=device))
            extend_vsizes.append(vocab_size + len(oov2idx))

        # Pad to max length
        max_len = max(len(t) for t in extend_list) if extend_list else 1
        extend_art = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(extend_list):
            extend_art[i, :len(seq)] = seq

        extend_vsize = max(extend_vsizes) if extend_vsizes else vocab_size

        return extend_art, extend_vsize


    def _convert_to_seneca_format(self, input_ids: torch.Tensor):
        """
        Convert flat input_ids to SENECA format (sentences + entities).
        
        Args:
            input_ids: [B, L] token sequence
        
        Returns:
            Dict with article_sents, sent_nums, clusters, cluster_nums, etc.
        """
        B, L = input_ids.size()
        device = input_ids.device
        
        # Get special tokens
        if hasattr(self.vocab, 'sep_idx'):
            sep_token = self.vocab.sep_idx
        elif hasattr(self.vocab, 'eos_idx'):
            sep_token = self.vocab.eos_idx
        else:
            sep_token = 3
        
        # Split into sentences
        article_sents_list = []
        sent_nums = []
        
        for b in range(B):
            seq = input_ids[b]
            sentences = []
            current_sent = []
            
            for token in seq:
                if token.item() == 0:  # padding
                    break
                
                current_sent.append(token.item())
                
                # End sentence at separator or max length
                if token.item() == sep_token or len(current_sent) >= 20:
                    if current_sent:
                        sentences.append(current_sent)
                        current_sent = []
            
            if current_sent:
                sentences.append(current_sent)
            
            if not sentences:
                sentences = [seq[:20].cpu().tolist()]
            
            article_sents_list.append(sentences)
            sent_nums.append(len(sentences))
        
        # Pad sentences
        max_sents = max(sent_nums)
        max_words = max(max(len(s) for s in sents) for sents in article_sents_list)
        
        article_sents = torch.zeros(B, max_sents, max_words, dtype=torch.long, device=device)
        for i, sents in enumerate(article_sents_list):
            for j, sent in enumerate(sents):
                article_sents[i, j, :len(sent)] = torch.tensor(sent, device=device)
        
        # Create pseudo-entity clusters (simplified)
        max_clusters = 3
        max_mentions = 5
        clusters = torch.zeros(B, max_clusters, max_mentions, dtype=torch.long, device=device)
        cluster_nums = torch.ones(B, dtype=torch.long) * max_clusters
        # --- Build flattened articles (concatenate sentences) ---
        flat_articles_list = []
        for i in range(B):
            sents = article_sents[i]  # [max_sents, max_words]
            nonzeros = []
            for j in range(sents.size(0)):
                sent = sents[j]
                nz = sent[sent != 0]
                if nz.numel() > 0:
                    nonzeros.append(nz)
            if nonzeros:
                flat = torch.cat(nonzeros, dim=0)
            else:
                # if nothing, pad with a single PAD token (0)
                flat = torch.tensor([0], dtype=torch.long, device=device)
            flat_articles_list.append(flat)

        # Pad flat articles to a rectangular tensor
        max_flat_len = max(x.size(0) for x in flat_articles_list)
        flat_articles = torch.zeros(B, max_flat_len, dtype=torch.long, device=device)
        for i, seq in enumerate(flat_articles_list):
            flat_articles[i, :seq.size(0)] = seq

        # Build extended vocab mapping for these flattened articles
        extend_art, extend_vsize = self._build_extended_for_batch(flat_articles)

        return {
            'article_sents': article_sents,
            'sent_nums': sent_nums,
            'clusters': clusters,
            'cluster_nums': cluster_nums,
            'extend_art': extend_art,
            'extend_vsize': extend_vsize
        }



    def _get_sents_from_indices(self, article_sents: torch.Tensor, 
                                indices: List[List[int]]) -> torch.Tensor:
        """
        Gather and flatten selected sentences by indices.
        
        Args:
            article_sents: [B, max_sents, max_words]
            indices: List[List[int]] sentence indices per sample
        
        Returns:
            [B, max_len] flattened selected sentences
        """
        B = article_sents.size(0)
        device = article_sents.device
        
        batch_extracted = []
        
        for i in range(B):
            if i < len(indices):
                idx_list = indices[i]
                if torch.is_tensor(idx_list):
                    idx_list = idx_list.cpu().tolist()
                elif isinstance(idx_list, int):
                    idx_list = [idx_list]
            else:
                idx_list = [0]
            
            # Clamp to valid range
            valid_indices = []
            for idx in idx_list:
                if isinstance(idx, (list, torch.Tensor)):
                    idx = idx[0] if len(idx) > 0 else 0
                if isinstance(idx, int) and 0 <= idx < article_sents.size(1):
                    valid_indices.append(idx)
            
            if not valid_indices:
                valid_indices = [0]
            
            # Gather and flatten sentences
            selected_sents = []
            for j in sorted(set(valid_indices)):
                sent = article_sents[i, j]
                non_padding = sent[sent != 0]
                if len(non_padding) > 0:
                    selected_sents.append(non_padding)
            
            if selected_sents:
                flattened = torch.cat(selected_sents, dim=0)
            else:
                flattened = article_sents[i, 0]
                flattened = flattened[flattened != 0]
            
            batch_extracted.append(flattened)
        
        # Pad to same length
        max_len = max(s.size(0) for s in batch_extracted) if batch_extracted else 1
        padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(batch_extracted):
            padded[i, :seq.size(0)] = seq
        
        return padded

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor,
                extend_art: Optional[torch.Tensor] = None,
                extend_vsize: Optional[int] = None):
        """
        Forward pass for training.
        
        Args:
            input_ids: [B, L] source tokens
            labels: [B, T] target tokens
            extend_art: optional extended vocab indices
            extend_vsize: optional extended vocab size
        
        Returns:
            summary_logits: [B, T, extend_vsize]
            total_loss: combined extraction + generation loss
        """
        B = input_ids.size(0)
        device = input_ids.device
        
        # Convert to SENECA format
        seneca_batch = self._convert_to_seneca_format(input_ids)
        
        if extend_art is None:
            extend_art = seneca_batch['extend_art']
        if extend_vsize is None:
            extend_vsize = seneca_batch['extend_vsize']
        
        # 1. Content selection with pseudo-labels
        max_extract = 3
        target_indices = torch.zeros(B, max_extract, dtype=torch.long, device=device)
        for i in range(B):
            num_sents = seneca_batch['sent_nums'][i]
            target_indices[i, :min(max_extract, num_sents)] = torch.arange(
                min(max_extract, num_sents), device=device
            )
        
        extraction_scores = self.extractor(
            article_sents=seneca_batch['article_sents'],
            sent_nums=torch.tensor(seneca_batch['sent_nums'], device=device),
            clusters=seneca_batch['clusters'],
            cluster_nums=seneca_batch['cluster_nums'],
            target_indices=target_indices
        )
        
        # Extraction loss
        ext_loss = self.extraction_loss(
            extraction_scores.view(-1, extraction_scores.size(-1)),
            target_indices.view(-1)
        )
        
        # 2. Abstract generation (GREEDY extraction during training)
        B, T, Ns = extraction_scores.size()
        flat_scores = extraction_scores.view(B, -1)
        _, flat_idx = flat_scores.topk(k=3, dim=-1)
        
        extracted_indices = []
        for i in range(B):
            sent_idxs = (flat_idx[i] % Ns).tolist()
            valid = [int(idx) for idx in sent_idxs if idx < seneca_batch['sent_nums'][i]]
            if not valid:
                valid = [0]
            
            # Remove duplicates
            seen = set()
            uniq = []
            for x in valid:
                if x not in seen:
                    uniq.append(x)
                    seen.add(x)
            extracted_indices.append(uniq)
        
        # Build extracted sentences
        extracted_sents = self._get_sents_from_indices(
            seneca_batch['article_sents'], extracted_indices
        )
        
        # Compute lengths
        art_lens = [int(extracted_sents[i].ne(0).sum().item()) for i in range(B)]
        
        # Build correct extended vocab for the extracted (flattened) sentences
        extracted_extend_art, extracted_extend_vsize = self._build_extended_for_batch(
            extracted_sents.to(device)
        )

        # Generate
        summary_logits = self.generator(
            article=extracted_sents,
            art_lens=art_lens,
            abstract=labels,
            extend_art=extracted_extend_art,
            extend_vsize=extracted_extend_vsize
        )

        
        # Generation loss
        gen_loss = self.generation_loss(
            summary_logits.view(-1, summary_logits.size(-1)),
            labels.view(-1)
        )
        
        # Combined loss
        total_loss = 0.3 * ext_loss + 0.7 * gen_loss
        
        return summary_logits, total_loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor,
                extend_art: Optional[torch.Tensor] = None,
                extend_vsize: Optional[int] = None,
                max_len: int = 120,
                min_len: int = 10) -> List[List[int]]:
        """
        Predict for inference (GREEDY generation).
        
        Args:
            input_ids: [B, L]
            extend_art: optional
            extend_vsize: optional
            max_len: max summary length
            min_len: min summary length
        
        Returns:
            List[List[int]] - generated token sequences
        """
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device
        
        # Convert to SENECA format
        seneca_batch = self._convert_to_seneca_format(input_ids)
        
        if extend_art is None:
            extend_art = seneca_batch['extend_art']
        if extend_vsize is None:
            extend_vsize = seneca_batch['extend_vsize']
        
        # 1. Extract sentences (GREEDY)
        k_extract = getattr(self.cfg.INFERENCE, 'TOP_K', 4)
        
        extracted_indices = self.extractor.extract_greedy(
            article_sents=seneca_batch['article_sents'],
            sent_nums=torch.tensor(seneca_batch['sent_nums'], device=device),
            clusters=seneca_batch['clusters'],
            cluster_nums=seneca_batch['cluster_nums'],
            k=k_extract
        )
        
        # Gather extracted sentences
        extracted_sents = self._get_sents_from_indices(
            seneca_batch['article_sents'], extracted_indices
        )

        art_lens = [
            extracted_sents[i].ne(0).sum().item() 
            for i in range(extracted_sents.size(0))
        ]

        # Build extended mapping for extracted_sents
        extracted_extend_art, extracted_extend_vsize = self._build_extended_for_batch(
            extracted_sents.to(device)
        )

        # 2. Generate summary (GREEDY)
        summaries = self.generator.greedy_generate(
            article=extracted_sents,
            art_lens=art_lens,
            extend_art=extracted_extend_art,
            extend_vsize=extracted_extend_vsize,
            go=getattr(self.vocab, 'bos_idx', 2),
            eos=getattr(self.vocab, 'eos_idx', 3),
            max_len=max_len,
            min_len=min_len
        )
        
        return summaries