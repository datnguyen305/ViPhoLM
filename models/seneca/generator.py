import torch
from torch import nn
from torch.nn import init
from .components import MultiLayerLSTMCells, CopyLSTMDecoder, _CopyLinear
from .utils import lstm_encoder, sequence_mean, len_mask, init_beam, pack_beam, next_search_beam

class AbstractGenerator(nn.Module):
    """
    Lớp Generator hoàn chỉnh với cơ chế copy.
    Logic chính được lấy từ CopySumm và lớp cha của nó là Seq2SeqSumm.
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout=0.0):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Encoder
        self._enc_lstm = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout, batch_first=True
        )
        
        # Các thành phần của Decoder
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_lstm_cells = MultiLayerLSTMCells(
            emb_dim + enc_out_dim, n_hidden, n_layer, dropout=dropout
        )
        
        # Lớp chiếu để khởi tạo trạng thái decoder
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        
        # Attention weights
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        
        # Lớp chiếu cuối cùng
        self._projection = nn.Sequential(
            nn.Linear(enc_out_dim + n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        
        # Lớp tính xác suất copy
        self._copy = _CopyLinear(enc_out_dim, n_hidden, emb_dim + enc_out_dim)
        
        # Decoder hoàn chỉnh
        self._decoder = CopyLSTMDecoder(
            self._copy, self._embedding, self._dec_lstm_cells,
            self._attn_wq, self._projection
        )

    def encode(self, article, art_lens=None):
        """Mã hóa các câu đầu vào."""
        # Giống hệt logic encode trong Seq2SeqSumm
        enc_art, (h, c) = lstm_encoder(article, self._enc_lstm, art_lens, embedding=self._embedding)
        
        if self._enc_lstm.bidirectional:
            h = torch.cat(h.chunk(2, dim=0), dim=2)
            c = torch.cat(c.chunk(2, dim=0), dim=2)
            
        init_h = torch.stack([self._dec_h(s) for s in h], dim=0)
        init_c = torch.stack([self._dec_c(s) for s in c], dim=0)
        init_dec_states = (init_h, init_c)
        
        attention = torch.matmul(enc_art, self._attn_wm)
        
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        
        return attention, (init_dec_states, init_attn_out)

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize):
        """Forward pass cho quá trình huấn luyện."""
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        
        decoder_input = (attention, mask, extend_art, extend_vsize)
        
        logits = self._decoder(decoder_input, init_dec_states, abstract)
        return logits

    # Thêm các phương thức greedy, sample, và batched_beamsearch từ CopySumm.py vào đây
    # Ví dụ:
    def batched_beamsearch(self, article, art_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, min_len=35, diverse=1.0):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        all_attention = (attention, mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            if t < min_len:
                force_not_stop = True
            else:
                force_not_stop = False
            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size, force_not_stop=force_not_stop)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, mask, extend_art, extend_vsize
                    ) = all_attention
                    masks = [mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, extend_art]
                    )
                    if masks:
                        mask = torch.stack(masks, dim=0)
                    else:
                        mask = None
                    attention = (
                        attention, mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs