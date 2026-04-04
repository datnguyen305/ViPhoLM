import torch 

def create_context_vector(
    encoder_hidden: torch.Tensor,\
    decoder_attn_weights: torch.Tensor\
    ):
    """
        encoder_hidden: (B, S_src, d_model)
        decoder_attn_weights: (B, S_trg, S_src)
        (B, S_trg, S_src) @ (B, S_src, d_model) -------> (B, S_trg, d_model)
        Returns: 
        context: (B, S_trg, d_model)
    """
    context = torch.bmm(decoder_attn_weights, encoder_hidden)
    return context
    