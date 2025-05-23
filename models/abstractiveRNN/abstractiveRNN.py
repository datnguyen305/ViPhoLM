import torch
from torch.nn import functional as F
import torch.nn as nn
import random
from vocabs.vocab import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from builders.model_builder import META_ARCHITECTURE

class Encoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size, padding_idx = vocab.pad_idx)
        self.pos_embedding = nn.Embedding(len(vocab.pos_itos),  config.hidden_size//4) # pos_emb_dim = 64
        self.ner_embedding = nn.Embedding(len(vocab.ner_itos),  config.hidden_size//4)  # ner_emb_dim = 64
        self.tfidf_embedding = nn.Linear(vocab.vocab_size, config.tfidf_emb_dim)
        self.rnn = nn.GRU(
            config.enc_emb_dim + config.pos_emb_dim + config.ner_emb_dim + config.tfidf_emb_dim, 
            config.enc_hid_dim,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear((config.emb_dim + config.pos_emb_dim + config.ner_emb_dim + config.tfidf_emb_dim)*2, config.dec_hid_dim)
        self.dropout = nn.Dropout(config.Dropout)

    def forward(self, input):
        # input: [batch_size, src_len]
        input_len = input.shape[1]
        pos_tags, ner_tags = self.vocab.encode_pos_ner(input, self.vocab.pos_tags, self.vocab.ner_tags)
        # pos_tags: [batch_size, src_len]
        # ner_tags: [batch_size, src_len]
        tfidf = self.vocab.get_tfidf_vector(input)
        # tfidf: [batch_size, src_len]

        word_embedded = self.dropout(self.embedding(input))
        # word_embedded: [batch_size, src_len, emb_dim]
        pos_embedded = self.dropout(self.pos_embedding(pos_tags))
        # pos_embedded: [batch_size, src_len, hidden_size//4]
        ner_embedded = self.dropout(self.ner_embedding(ner_tags))
        # ner_embedded: [batch_size, src_len, hidden_size//4]
        tfidf_embedded = self.dropout(self.tfidf_embedding(self.vocab.tfidf_values))
        # tfidf_embedded: [batch_size, tfidf_emb_dim]
        tfidf_embedded = tfidf_embedded.unsqueeze(1).repeat(1, input_len, 1)
        # tfidf_embedded: [batch_size, src_len, tfidf_emb_dim]

        concatenated_input = torch.tanh(torch.cat((word_embedded, pos_embedded, ner_embedded, tfidf_embedded), dim=-1))
        # concatenated_input: [batch_size, src_len, emb_dim + pos_emb_dim + ner_emb_dim + tfidf_emb_dim]
        embedded = self.dropout(concatenated_input)
        # embedded: [batch_size, src_len, emb_dim + pos_emb_dim + ner_emb_dim + tfidf_emb_dim]
        packed_embedded = pack_padded_sequence(embedded, input_len.cpu() , batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        # hidden: [num_layers * num_directions, batch_size, emb_dim + pos_emb_dim + ner_emb_dim + tfidf_emb_dim]
        outputs,_ = pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, src_len, num_directions * emb_dim + pos_emb_dim + ner_emb_dim + tfidf_emb_dim]

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=-1)))
        # hidden: [batch_size, dec_hid_dim aka hidden_size*2]
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = nn.Linear((config.enc_hid_dim * 2) + config.dec_hid_dim, config.dec_hid_dim)
        self.v = nn.Linear(config.dec_hid_dim, 1, bias=False)
        self.coverage = nn.Linear(1, config.dec_hid_dim)

    def forward(self, hidden, encoder_outputs, coverage_vector):
        # hidden: [batch_size, dec_hid_dim] 
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]

        src_len = encoder_outputs.shape[1]
        coverage_vector = self.coverage(coverage_vector.unsqueeze(-1))
        # coverage_vector: [batch_size, src_len, dec_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs, coverage_vector.unsqueezed(2)), dim=2)))
        # energy: [batch_size, src_len, dec_hid_dim]  
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        attention = F.softmax(attention, dim=1)
        new_coverage = coverage_vector + attention
        return attention, new_coverage
    
class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        
        self.output_dim = vocab.vocab_size
        self.attention = Attention()
        self.device = config.device
        self.embedding = nn.Embedding(self.output_dim, config.dec_emb_dim, padding_idx=vocab.pad_idx)
        self.rnn = nn.GRU((config.enc_hid_dim * 2) + config.dec_emb_dim, config.dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((config.enc_hid_dim * 2) + config.dec_hid_dim + config.dec_emb_dim, self.output_dim)
        self.p_gen = nn.Linear((config.enc_hid_dim * 2) + config.dec_hid_dim + config.dec_emb_dim, 1)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input, hidden , encoder_outputs, src_words, coverage_vector):
        # input: [batch_size]: y_i-1
        # hidden: [batch_size, dec_hid_dim]: s_i-1
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]: h_j

        input = input.unsqueeze(1)
        # input: [batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, dec_emb_dim]
        a, new_coverage = self.attention(hidden, encoder_outputs, coverage_vector)
        # a [batch_size, src_len]
        a = a.unsqueeze(1)
        # a [batch_size, 1, src_len]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        weight = a.bmm(encoder_outputs)
        # weight: [batch_size, 1, enc_hid_dim * 2]
        rnn_input = torch.cat((embedded,weight), dim=2)
        # rnn_input: [batch_size, 1, dec_emb_dim + enc_hid_dim * 2]
        # hidden: [batch_size, dec_hid_dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: [batch_size, 1, dec_hid_dim]
        # hidden: [1, batch_size, dec_hid_dim]
        # weights: [batch_size, 1, enc_hid_dim * 2]
        prediction = self.fc_out(torch.cat((output, weight, embedded), dim=2))
        # prediction: [batch_size, 1, output_dim]

        p_gen_input = torch.cat((output.squeeze(1), hidden.squeeze(0), embedded.squeeze(1)), dim=-1)
        # p_gen_input: [batch_size, dec_hid_dim + dec_hid_dim + dec_emb_dim]
        p_gen = torch.sigmoid(self.p_gen(p_gen_input))
        # p_gen: [batch_size, 1]
        vocab_dist = F.softmax(prediction.squeeze(1), dim=-1) * p_gen
        copy_dist = a * (1 - p_gen)
        # copy_dist: [batch_size, src_len]
        final_dist = torch.zeros(input.shape[0], self.vocab.vocab_size).to(self.device)
        # final_dist: [batch_size, vocab_size]
        final_dist.scatter_add_(1, src_words, copy_dist)  # Copy từ input
        final_dist[:, :vocab_dist.size(1)] += vocab_dist

        return final_dist, hidden.squeeze(0), a.squeeze(1), new_coverage

@META_ARCHITECTURE.register()
class abstractiveRNN_model(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        
        self.encoder = Encoder(config, vocab)
        self.decoder = Decoder(config, vocab)
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2 # + 2 for bos and eos tokens

        self.device = config.device
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        
    def forward(self, src, trg): # teacher forcing = 0.5
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        teacher_forcing_ratio = 0.5
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.vocab.vocab_size
        
        coverage_vector = torch.zeros(batch_size, src.shape[1]).to(self.device)
        total_loss = 0
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[:, 0]  # First input is <sos> token
        for t in range(self.MAX_LENGTH):
            output, hidden, attn, coverage_vector = self.decoder(input, hidden, encoder_outputs, src, coverage_vector)
            outputs[:, t] = output
            cov_loss = torch.sum(torch.min(attn, coverage_vector), dim=1)  # [batch_size]
            loss = self.loss_fn(output, trg[:, t]) + 0.1 * cov_loss.mean()  # Lambda = 0.1
            total_loss += loss
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
                    
        return outputs, total_loss
    
    def predict(self, src, max_len=50):
        """Generate translation without teacher forcing"""
        with torch.no_grad():
            batch_size = src.shape[0]
            src_len = src.shape[1]
            
            encoder_outputs, hidden = self.encoder(src, src_len)
            
            outputs = torch.zeros(batch_size, max_len, self.vocab.vocab_size).to(self.device)
            input = torch.tensor([self.vocab.bos_idx] * batch_size).to(self.device)
            
            for t in range(self.MAX_LENGTH):
                output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
                outputs[:, t] = output
                input = output.argmax(1)
                
                # Stop if all sequences generated <eos>
                if (input == self.vocab.eos_idx).all():
                    break
            
            return outputs.argmax(2)  # [batch_size, max_len]
