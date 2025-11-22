from torch import nn

from models.transformer_hepos.embedding.positional_encoding import PositionalEncoding
from models.transformer_hepos.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        max_len = min(x.shape[1], self.pos_emb.max_len)  # Giới hạn độ dài
        x = x[:, :max_len]  # Cắt input_ids nếu cần

        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x[:, :max_len])  # Tạo pos_emb theo chiều dài thực tế
        return self.drop_out(tok_emb + pos_emb)
