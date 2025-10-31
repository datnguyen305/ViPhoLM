import torch
from torch.nn import functional as F
import torch.nn as nn
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

class HierarchicalEncoder(nn.Module):
    """
    Input: 
        input: (B, S_s, S_w)  # batch of documents
            B: batch size
            S_s: number of sentences
            S_w: words per sentence
    Output:
        word_outputs: (B, S_s, S_w, hidden_size)  # word-level encoder outputs
        sent_outputs: (B, S_s, hidden_size)       # sentence-level encoder outputs
        hidden: (B, hidden_size)                    # final document representation
    """
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        # Word-level encoder
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.word_gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            batch_first=True,
            bidirectional=config.encoder.bidirectional,
            dropout=config.dropout,
            num_layers=config.layer_dim,
            device=config.device
        )
        
        # Sentence-level encoder
        self.sent_gru = nn.GRU(
            config.hidden_size * 2 if config.encoder.bidirectional else config.hidden_size,
            config.hidden_size,
            batch_first=True,
            bidirectional=config.encoder.bidirectional,
            dropout=config.dropout,
            num_layers=config.layer_dim,
            device=config.device
        )
        
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, input):
        """
        Inputs:
            input: (B, S_s, S_w)
        Outputs:
            word_outputs: (B, S_s, S_w, hidden_size)  # word-level encoder outputs
            sent_outputs: (B, S_s, hidden_size)       # sentence-level encoder outputs
            hidden: (B, hidden_size)                    # final document representation
        """
        # Input shape: (B, S_s, S_w)
        B, S_s, S_w = input.size()
        
        # 1. Word-level encoding
        # Reshape for word processing
        x = input.view(B * S_s, S_w)  # (B*S_s, S_w)
        x = self.embedding(x)          # (B*S_s, S_w, H)

        word_outputs, word_hidden = self.word_gru(x)
        # word_outputs: (B*S_s, S_w, H*2)
        # word_hidden: (num_layers*2, B*S_s, H)

        # 2. Sentence-level encoding
        # Use last hidden state of each sentence
        sent_input = word_hidden.view(B, S_s, -1)  # (B, S_s, H*2)

        sent_outputs, sent_hidden = self.sent_gru(sent_input)
        # sent_outputs: (B, S_s, H*2)
        # sent_hidden: (num_layers*2, B, H)

        # 3. Get final document representation
        hidden = F.relu(self.fc(torch.cat(
            (sent_hidden[-2,:,:], sent_hidden[-1,:,:]), 
            dim=1
        )))  # (B, H)
        
        # Reshape word outputs back to hierarchical form
        word_outputs = word_outputs.view(B, S_s, S_w, -1) # (B, S_s, S_w, H*2)  
        word_outputs = self.fc(word_outputs) # (B, S_s, S_w, H)

        sent_outputs = self.fc(sent_outputs) # (B, S_s, H)
        return word_outputs, sent_outputs, hidden

class Encoder(nn.Module):
    """
    Input: 
        input: (B, S)  # batch of input sequences
    Output: 
        output: (B, S, hidden_size)  # encoder outputs for all time steps
        hidden: (B, hidden_size)  # final hidden state to be used as context
    """
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True, #(B, S, hidden_size)
            dropout=config.dropout,
            bidirectional=config.bidirectional
        )
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size) #bidirectional
    def forward(self, input):
        embedded = self.embedding(input) # (B, S, hidden_size)
        output, hidden = self.gru(embedded)
        #hidden:(num_layers * num_directions, B, hidden_size), a = [a->, a<-]
        
        #Thử dùng relu thay vì tanh
        hidden = F.relu(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) #hidden: (num_layers * num_directions, B, hidden_size) -> (B, hidden_size*2) -> (B, hidden_size)
        #Ghép lại thành 1 chiều dữ liệu để làm context ban đầu cho decoder

        return output, hidden

class HierarchicalDecoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size * 2,  # embedding + context
            config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False
        )
        # Use hierarchical attention instead of regular attention
        self.attention = HierarchicalAttention(config)
        self.W1 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.W2 = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.b2 = nn.Parameter(torch.zeros(vocab.vocab_size))
        self.pointer = PointerGeneratorDecoder(config, vocab)

    def forward_step(self, input, hidden, word_outputs, sent_outputs, src_tokens):
        """
        Inputs:
            input: (B, 1) - Current input token
            hidden: (B, H) - Previous decoder hidden state
            word_outputs: (B, S_s, S_w, H) - Word-level encoder outputs
            sent_outputs: (B, S_s, H) - Sentence-level encoder outputs
            src_tokens: (B, S_s*S_w) - Flattened source tokens
        Outputs:
            P_final: (B, vocab_size) - Final probability distribution over vocabulary
            hidden: (B, H) - Updated decoder hidden state
            attn_dist: (B, S_s*S_w) - Attention distribution over source
            p_gen: (B, 1) - Generation probability
        """
        # 1. Embed input
        embedded = self.embedding(input)  # (B, 1, H)
        
        # 2. Calculate hierarchical attention
        context_vector, attn_dist = self.attention(
            word_outputs,
            sent_outputs, 
            hidden
        )
        context_vector = context_vector.unsqueeze(1)  # (B, 1, H)
        
        # 3. Concatenate embedding and context
        gru_input = torch.cat((embedded, context_vector), dim=-1)  # (B, 1, H*2)
        
        # 4. GRU step
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        
        # 5. Calculate vocabulary distribution
        concat = torch.cat((hidden, context_vector.squeeze(1)), dim=-1)  # (B, H*2)
        P_vocab = F.softmax(
            self.W2(F.tanh(self.W1(concat))) + self.b2, 
            dim=-1
        )  # (B, vocab_size)
        
        # 6. Get generation probability
        p_gen = self.pointer(hidden, output, word_outputs.view(word_outputs.size(0), -1, word_outputs.size(-1)))
        
        # 7. Calculate final distribution
        P_final = p_gen * P_vocab
        
        # 8. Add copy distribution
        copy_dist = torch.zeros_like(P_final)
        copy_dist.scatter_add_(
            1,
            src_tokens,
            (1 - p_gen) * attn_dist
        )
        P_final = P_final + copy_dist
        
        return P_final, hidden, attn_dist, p_gen

    def forward(self, target=None, hidden=None, word_outputs=None, sent_outputs=None, src_tokens=None, max_len=50):
        """
        Inputs:
            target: (B, T) - Target sequence (optional, used during training)
            hidden: (B, H) - Initial decoder hidden state
            word_outputs: (B, S_s, S_w, H) - Word-level encoder outputs
            sent_outputs: (B, S_s, H) - Sentence-level encoder outputs
            src_tokens: (B, S) - Source tokens for copy mechanism
            max_len: int - Maximum length for inference
        Outputs:
            outputs: (B, T, vocab_size) - Decoder outputs
            hidden: (B, H) - Final decoder hidden state
            attn_dist: (B, S) - Attention distribution over source
            p_gen: (B, 1) - Generation probability
        """
        batch_size = word_outputs.size(0)
        
        # Initialize outputs
        outputs = []
        attn_dists = []
        p_gens = []
        decoder_input = torch.full(
            (batch_size, 1),
            self.vocab.bos_idx,
            device=word_outputs.device
        )
        
        # Training mode with teacher forcing
        if self.training and target is not None:
            for t in range(target.size(1)):
                output, hidden, attn_dist, p_gen = self.forward_step(
                    decoder_input, hidden, word_outputs, sent_outputs, src_tokens
                )
                outputs.append(output)
                attn_dists.append(attn_dist)
                p_gens.append(p_gen)
                decoder_input = target[:, t].unsqueeze(1)
            
            outputs = torch.stack(outputs, dim=1)
            attn_dists = torch.stack(attn_dists, dim=1)
            p_gens = torch.stack(p_gens, dim=1)
            return outputs, hidden, attn_dists, p_gens
                
        # Inference mode
        else:
            for t in range(max_len):
                # Forward pass through decoder
                output, hidden, attn_dist, p_gen = self.forward_step(
                    decoder_input, hidden, word_outputs, sent_outputs, src_tokens
                )
                outputs.append(output)
                
                # Next input is current prediction
                decoder_input = output.argmax(dim=-1).unsqueeze(1)
                
                # Stop if all sequences have generated EOS
                if (decoder_input == self.vocab.eos_idx).all():
                    break
                    
        # Stack all outputs
        outputs = torch.stack(outputs, dim=1)  # (B, T, vocab_size)
        
        return outputs, hidden, attn_dist, p_gen

class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab.vocab_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size * 2, 
            config.hidden_size,
            num_layers=config.layer_dim,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False
        )
        self.attention = BahdanauAttention(config)
        self.W1 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.W2 = nn.Linear(config.hidden_size, vocab.vocab_size)
        self.b2 = nn.Parameter(torch.zeros(vocab.vocab_size))
        self.pointer = PointerGeneratorDecoder(config, vocab)

    def forward_step(self, input, hidden, encoder_outputs, src_tokens):
        """
        Inputs: 
            input: (B, 1) - current input token
            hidden: (B, H) - previous decoder hidden state
            encoder_outputs: (B, S, H) - encoder outputs
            src_tokens: (B, S) - source tokens for copy mechanism
        Outputs:
            P_final: (B, vocab_size) - final probability distribution over vocabulary
            hidden: (B, H) - updated decoder hidden state
            attn_dist: (B, S) - attention distribution over source
            p_gen: (B, 1) - generation probability
        """
        # 1. Embed input
        embedded = self.embedding(input)  # (B, 1, H)
        # 2. Calculate attention
        context_vector, attn_dist = self.attention(hidden, encoder_outputs)
        context_vector = context_vector.unsqueeze(1)  # (B, 1, H)
        # 3. Concatenate embedding and context
        gru_input = torch.cat((embedded, context_vector), dim=-1)  # (B, 1, H*2)
        # 4. GRU step
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        # 5. Calculate vocabulary distribution

        
        concat = torch.cat((hidden, context_vector.squeeze(1)), dim=-1)  # (B, H*2)
        P_vocab = F.softmax(
            self.W2(F.tanh(self.W1(concat))) + self.b2, 
            dim=-1
        )  # (B, vocab_size)
        # 6. Get generation probability
        p_gen = self.pointer(hidden, output, encoder_outputs)
        # 7. Calculate final distribution
        P_final = p_gen * P_vocab
        # 8. Add copy distribution
        copy_dist = torch.zeros_like(P_final)
        copy_dist.scatter_add_(
            1,
            src_tokens,
            (1 - p_gen) * attn_dist
        )
        P_final = P_final + copy_dist
        return P_final, hidden, attn_dist, p_gen

    def forward(self, target=None, hidden=None, encoder_outputs=None, src_tokens=None, max_len=50):
        """
        Inputs:
            target: (B, T) - Target sequence (optional, used during training)
            hidden: (B, H) - Initial decoder hidden state
            encoder_outputs: (B, S, H) - Encoder outputs
            src_tokens: (B, S) - Source tokens for copy mechanism
            max_len: int - Maximum length for inference
        Outputs:
            outputs: (B, T, vocab_size) - Decoder outputs
            hidden: (B, H) - Final decoder hidden state
            attn_dist: (B, S) - Attention distribution over source
            p_gen: (B, 1) - Generation probability
        """
        batch_size = encoder_outputs.size(0)
        
        # Initialize outputs
        outputs = []
        decoder_input = torch.full(
            (batch_size, 1),
            self.vocab.bos_idx,
            device=encoder_outputs.device
        )
        
        # Training mode with teacher forcing
        if self.training and target is not None:
            target_length = target.size(1)
            
            for t in range(target_length):
                # Forward pass through decoder
                output, hidden, attn_dist, p_gen = self.forward_step(
                    decoder_input, hidden, encoder_outputs, src_tokens
                )
                outputs.append(output)
                
                # Teacher forcing: next input is current target
                decoder_input = target[:, t].unsqueeze(1)
                
        # Inference mode
        else:
            for t in range(max_len):
                # Forward pass through decoder
                output, hidden, attn_dist, p_gen = self.forward_step(
                    decoder_input, hidden, encoder_outputs, src_tokens
                )
                outputs.append(output)
                
                # Next input is current prediction
                decoder_input = output.argmax(dim=-1).unsqueeze(1)
                
                # Stop if all sequences have generated EOS
                if (decoder_input == self.vocab.eos_idx).all():
                    break
                    
        # Stack all outputs
        outputs = torch.stack(outputs, dim=1)  # (B, T, vocab_size)
        
        return outputs, hidden, attn_dist, p_gen

class BahdanauAttention(nn.Module): # Attention Bahdanau-style
    def __init__(self, config):
        super().__init__()
        # Word-Level attention
        self.W_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_s = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_a = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, prev_decoder_hidden, encoder_outputs): 
        """
        Inputs:
            prev_decoder_hidden: (B, hidden_size)
            encoder_outputs: (B, S, hidden_size)
        Outputs:
            context_vector: (B, hidden_size)
            attention_weights: (B, S)
        """
        Wi_hi = self.W_h(encoder_outputs)  # (B, S, hidden_size)
        Ws_prev_s = self.W_s(prev_decoder_hidden)  #(B, hidden_size)
        E_ti = self.v_a(torch.tanh(Wi_hi + Ws_prev_s.unsqueeze(1))).squeeze(-1)  # (B, S)
        A_ti = F.softmax(E_ti, dim=1) #(B, S)
        C_t = F.bmm(A_ti.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, hidden_size)
        return C_t, A_ti  # context_vector, attention_weights

def rescale_attn(word_attn, sent_attn, B, S_s, S_w):
    """
    Calculate final attention weights over words given word-level and sentence-level attention.
    Inputs:
        word_attn: (B, S_s*S_w)  # word-level attention weights
        sent_attn: (B, S_s)      # sentence-level attention weights
    Outputs:
        final_attn: (B, S_s, S_w)  # final attention weights over words
    """

    word_attn = word_attn.view(B, S_s, S_w)  # (B, S_s, S_w)
    final_attn = word_attn * sent_attn.unsqueeze(-1) # (B, S_s, S_w) * (B, S_s, 1) -> (B, S_s, S_w)
    denominator = final_attn.sum(dim=[1,2], keepdim=True)  # (B, 1, 1)
    final_attn_scaled = final_attn / (denominator + 1e-12)  # (B, S_s, S_w), +1e-12 to avoid division by zero
    return final_attn_scaled

class HierarchicalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sentence_attn = BahdanauAttention(config)
        self.word_attn = BahdanauAttention(config)
    def forward(self, word_encoder_output, sentenct_encoder_output, prev_decoder_hidden):
        """
        Inputs:
            word_encoder_output: (B, S_s, S_w, hidden_size)
            sentenct_encoder_output: (B, S_s, hidden_size)
            prev_decoder_hidden: (B, hidden_size)
        Outputs: 
            final_context: (B, hidden_size)
            final_attn: (B, S_s*S_w)
        """
        B, S_s, S_w, H = word_encoder_output.size()
        word_encoder_output = word_encoder_output.view(B, S_s * S_w, H)  # (B, S_s*S_w, hidden_size)
        word_context, word_attn = self.word_attn(prev_decoder_hidden, word_encoder_output)
        # word_context: (B, hidden_size)
        # word_attn: (B, S_s*S_w)

        sent_context, sent_attn = self.sentence_attn(prev_decoder_hidden, sentenct_encoder_output)
        # sent_context: (B, hidden_size)
        # sent_attn: (B, S_s)

        final_attn = rescale_attn(word_attn, sent_attn, B, S_s, S_w)  # (B, S_s, S_w)
        final_context = torch.bmm(final_attn.view(B, 1, S_s * S_w), word_encoder_output).squeeze(1)  # (B, hidden_size)
        final_attn = final_attn.view(B, -1) # (B, S)
        return final_context, final_attn  # (B, hidden_size), (B, S)
    
class PointerGeneratorDecoder(nn.Module):
    """
    Inputs: 
        decoder_hidden: (B, hidden_size) __ h 
        decoder_output: (B, S, hidden_size) __ E
        encoder_outputs: (B, S, hidden_size) __ c
    Outputs:
        p_gen: (B, 1)
    """
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.Ws_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.Ws_e = nn.Linear(config.hidden_size, config.hidden_size)
        self.Ws_c = nn.Linear(config.hidden_size, config.hidden_size)
        self.va = nn.Linear(config.hidden_size, 1, bias=False)
        self.ba = nn.Parameter(torch.zeros(config.hidden_size)) #bias for attention, shape: (hidden_size,)
        self.attention = BahdanauAttention(config)
    def forward(self, decoder_hidden, decoder_output, encoder_outputs): 
        decoder_output = decoder_output.squeeze(1)  # (B, hidden_size)
        Ws_h_i = self.Ws_h(decoder_hidden)  # (B, hidden_size)
        Ws_e_i = self.Ws_e(decoder_output)  #(B, hidden_size)
        Ws_c_i, _ = self.Ws_c(self.attention(decoder_hidden, encoder_outputs)) # (B, hidden_size)
        scores = F.tanh(Ws_h_i + Ws_e_i + Ws_c_i + self.ba)  # (B, hidden_size), Tại sao có hàm tanh? do NN hoạt động dựa trên non-linearity nhưng với phép cộng nó chỉ đang là linear nên phải thêm activation func là tanh
        energy = self.va(scores)  # (B, 1)
        p_gen = torch.sigmoid(energy)  # (B, 1)
        return p_gen  # (B, 1)
    
def calculate_g(target_idx, vocab_size):
    """
    Calculates g (generate/copy indicator) for pointer-generator loss
    
    Args:
        target_idx: (B,) - Target word indices
        vocab_size: int - Size of vocabulary
        
    Returns:
        g: (B,) - Binary tensor, 1 if target in vocab, 0 if needs copying
    """
    return (target_idx < vocab_size).float()  # Convert bool to float

class PointerGeneratorLoss(nn.Module):
    """
    Loss module for Pointer-Generator Network.
    Implements the log-likelihood loss combining generation and copying probabilities.
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def calculate_g(self, target_idx, vocab_size):
        """
        Calculates binary indicators for generation vs copying
        
        Args:
            target_idx: (B,) - Target word indices
            vocab_size: int - Size of vocabulary
            
        Returns:
            g: (B,) - Binary tensor, 1 if target in vocab, 0 if needs copying
        """
        return (target_idx < vocab_size).float()

    def forward(self, P_vocab, attn_dist, p_gen, target_idx, copy_idx):
        """
        Forward pass to calculate loss
        
        Args:
            P_vocab: (B, vocab_size) - Generation probabilities over vocabulary
            attn_dist: (B, src_len) - Attention distribution over source
            p_gen: (B, 1) - Generation probability
            target_idx: (B,) - Target word indices 
            copy_idx: (B,) - Source positions for copying
            
        Returns:
            loss: scalar - Mean loss over batch
        """
        # Get generation probabilities for target words
        P_vocab_selected = torch.gather(
            P_vocab, 1, 
            target_idx.unsqueeze(1)
        ).squeeze(1)  # (B,)
        
        # Get copy probabilities from attention
        P_copy_selected = torch.gather(
            attn_dist, 1,
            copy_idx.unsqueeze(1)
        ).squeeze(1)  # (B,)

        # Calculate generation indicators
        g = self.calculate_g(target_idx, P_vocab.size(1))  # (B,)

        # Calculate loss terms
        gen_term = g * torch.log(
            p_gen.squeeze(1) * P_vocab_selected + self.eps
        )
        copy_term = (1 - g) * torch.log(
            (1 - p_gen.squeeze(1)) * P_copy_selected + self.eps
        )

        # Combine terms and take mean
        loss = -(gen_term + copy_term)
        return loss.mean()

def get_copy_indices(src_tokens, target_tokens, vocab_size):
    """
    Find positions in source sequence for copying target tokens
    
    Args:
        src_tokens: (B, S) - Source sequence 
        target_tokens: (B, T) - Target sequence
        vocab_size: int - Size of vocabulary
    Returns:
        copy_idx: (B, T) - Source positions for copying each target token
    """
    batch_size, src_len = src_tokens.size()
    _, tgt_len = target_tokens.size()
    device = src_tokens.device
    
    # Initialize with zeros
    copy_idx = torch.zeros_like(target_tokens)
    
    # For each position in target
    for b in range(batch_size):
        for t in range(tgt_len):
            tgt_token = target_tokens[b, t].item()
            # If token is OOV
            if tgt_token >= vocab_size:
                # Find matching position in source
                matches = (src_tokens[b] == tgt_token).nonzero()
                if len(matches) > 0:
                    # Take first match position
                    copy_idx[b, t] = matches[0]
                
    return copy_idx

@META_ARCHITECTURE.register()
class AbstractiveTXModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.MAX_LENGTH = vocab.max_sentence_length + 2
        self.encoder = HierarchicalEncoder(config, vocab)
        self.decoder = HierarchicalDecoder(config, vocab)
        self.loss = PointerGeneratorLoss()
        self.device = config.device

        self.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        """
        Inputs: 
            src: (B, S_s, S_w) - Source documents
            trg: (B, T) - Target summaries
        """
        # 1. Encode source document hierarchically
        word_outputs, sent_outputs, hidden = self.encoder(src)
        
        # 2. Decode with hierarchical attention
        decoder_outputs, _, attn_dists, p_gens = self.decoder(
            target=trg,
            hidden=hidden, 
            word_outputs=word_outputs,
            sent_outputs=sent_outputs,
            src_tokens=src.view(src.size(0), -1)  # Flatten source tokens
        )
        # 3. Calculate copy indices for loss
        copy_idx = get_copy_indices(
            src.view(src.size(0), -1),
            trg,
            self.vocab.vocab_size
        )
        total_loss = 0
        for t in range(trg.size(1)):
            step_loss = self.loss(
                P_vocab=decoder_outputs[:, t],
                attn_dist=attn_dists[:, t],
                p_gen=p_gens[:, t],
                target_idx=trg[:, t],
                copy_idx=copy_idx[:, t]
            )
            total_loss += step_loss
        return decoder_outputs, total_loss / trg.size(1)

    def predict(self, src: torch.Tensor):
        # 1. Encode source document
        word_outputs, sent_outputs, hidden = self.encoder(src)
        batch_size = src.size(0)
        decoder_input = torch.full(
            (batch_size, 1),
            self.vocab.bos_idx,
            device=self.device
        )
        
        outputs = []

        for _ in range(self.MAX_LENGTH):
            output, hidden, attn_dist, p_gen = self.decoder.forward_step(
                decoder_input, hidden, word_outputs, sent_outputs,
                src.view(src.size(0), -1)
            )
            outputs.append(output)
            decoder_input = output.argmax(dim=-1).unsqueeze(1)
            if (decoder_input == self.vocab.eos_idx).all():
                break
        return torch.cat(outputs, dim=1)