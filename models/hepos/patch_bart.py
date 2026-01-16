from models.hepos.bart_attention import HEPOSBartAttention


def patch_bart_with_hepos(model, stride):
    """
    Replace BART decoder cross-attention with HEPOS attention
    """
    for layer in model.model.decoder.layers:
        old_attn = layer.encoder_attn

        layer.encoder_attn = HEPOSBartAttention(
            embed_dim=old_attn.embed_dim,
            num_heads=old_attn.num_heads,
            stride=stride,
            dropout=old_attn.dropout,
        )
