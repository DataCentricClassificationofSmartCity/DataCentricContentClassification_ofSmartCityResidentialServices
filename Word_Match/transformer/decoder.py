import torch.nn as nn
from src.training.transformer_layers import MultiHeadAttention, PositionwiseFeedForward

# credit to "Yu-Hsiang Huang"


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,position_encoding_layer, n_layers, n_groups, n_head, d_features, max_seq_length,
                 d_meta, d_k=None, d_v=None, dropout=0.1, d_bottleneck=256):

        super().__init__()

        self.position_enc = position_encoding_layer(d_features=d_features, max_length=max_seq_length, d_meta=d_meta)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_features, d_bottleneck, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_features, eps=1e-6)
        self.d_features = d_features

    def forward(self, dec_input, dec_pos, enc_output, trg_mask, src_mask=None):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward

        dec_output = self.dropout(self.position_enc(dec_pos) + dec_input)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, dec_slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn]
            dec_enc_attn_list += [dec_enc_attn]

        return dec_output, dec_slf_attn_list, dec_enc_attn_list




# class DecoderLayer(nn.Module):
#     def __init__(self, multi_head_attn, position_wise_ff):
#         super().__init__()
#
#         self.self_attn = multi_head_attn
#         self.bottleneck = position_wise_ff
#
#     def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None):
#         '''
#             Arguments:
#                 enc_input {Tensor, shape [batch, length, d_features]} -- input
#                 non_pad_mask {Tensor, shape [batch, length, 1]} -- index of which position in a sequence is a padding
#                 slf_attn_mask {Tensor, shape [batch, length, length]} -- self attention mask
#
#             Returns:
#                 enc_output {Tensor, shape [batch, length, d_features]} -- output
#                 encoder_self_attn {n_head * batch, length, length} -- n_head self attention matrices
#         '''
#
#         enc_output, encoder_self_attn = self.self_attn(dec_input, enc_output, enc_output, key_padding_mask=non_pad_mask,
#                                                        mask=slf_attn_mask)
#         enc_output = self.bottleneck(enc_output)
#         return enc_output, encoder_self_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_features, d_bottleneck, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_features, d_k, d_v, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_features, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_features, d_bottleneck, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            dec_slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=dec_slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

