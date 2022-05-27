import numpy as np
import torch
import torch.nn as nn


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''
        Arguments:
            n_position {Int} -- the maximum position
            d_hid {Int} -- the dimension of the embedding
            padding_idx -- padding symbol

        Returns:
            sinusoid_table {Tensor, shape: [n_position+1, d_hid]} -- sinusoid encoding table
    '''
    n_position += 1

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_non_pad_mask(seq, padding_idx=0):
    '''
        Arguments:
            seq {batch, length} -- the word index of the sequence
            padding_idx -- padding symbol

        Returns:
            mask {Tensor, shape: [batch, length, 1]} -- non padding mask, 1 means not padding position while 0 means padding position
    '''
    assert seq.dim() == 2
    mask = seq.ne(padding_idx).type(torch.float)
    return mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, padding_idx=0):
    '''
        For masking out the padding part of key sequence.
        Arguments:
            seq_k {Tensor, shape [batch, len_k]} -- key sequence
            seq_q {Tensor, shape [batch, len_q]} -- query sequence

        Returns:
            padding_mask {Tensor, shape [batch, len_q, len_k]} -- mask matrix
            key mask [batch, k] -> expand q times [batch, len_q, len_k]

    '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask.eq(1)


class Encoder(nn.Module):
    ''' A encoder models with self attention mechanism. '''

    def __init__(self, position_encoding_layer, n_layers, n_groups, n_head, d_features, max_seq_length,
                 d_meta, d_k=None, d_v=None, dropout=0.1, d_bottleneck=256):
        super().__init__()
        assert n_layers % n_groups == 0, 'n_groups must be factor of n_layers'

        self.position_enc = position_encoding_layer(d_features=d_features, max_length=max_seq_length, d_meta=d_meta)

        # layers in one group share weights together
        layer_num_per_group = int(n_layers / n_groups)
        self.group_stack = nn.ModuleList([EncoderLayerGroup(layer_num_per_group, d_features, n_head, d_k, d_v,
                                                            dropout, d_bottleneck)
                                          for _ in range(n_groups)])

    def forward(self, feature_sequence, position, non_pad_mask=None, slf_attn_mask=None):
        '''
            Arguments:
                input_feature_sequence {Tensor, shape [batch, max_sequence_length, d_features]} -- input feature sequence
                position {Tensor, shape [batch, max_sequence_length (, d_meta)]} -- input feature position sequence

            Returns:
                enc_output {Tensor, shape [batch, max_sequence_length, d_features]} -- encoder output (representation)
                encoder_self_attn_list {List, length: n_layers} -- encoder self attention list,
                each element is a Tensor with shape [n_head * batch, max_sequence_length, max_sequence_length]
        '''

        encoder_self_attn_list = []

        # Add position information at the beginning
        enc_output = feature_sequence + self.position_enc(position)

        for group in self.group_stack:
            enc_output, encoder_self_attn = group(enc_output, non_pad_mask=non_pad_mask,
                                                  slf_attn_mask=slf_attn_mask)
            encoder_self_attn_list.append(encoder_self_attn)
        return enc_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    def __init__(self, multi_head_attn, position_wise_ff):
        super().__init__()

        self.self_attn = multi_head_attn
        self.bottleneck = position_wise_ff

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        '''
            Arguments:
                enc_input {Tensor, shape [batch, length, d_features]} -- input
                non_pad_mask {Tensor, shape [batch, length, 1]} -- index of which position in a sequence is a padding
                slf_attn_mask {Tensor, shape [batch, length, length]} -- self attention mask

            Returns:
                enc_output {Tensor, shape [batch, length, d_features]} -- output
                encoder_self_attn {n_head * batch, length, length} -- n_head self attention matrices
        '''

        enc_output, encoder_self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.bottleneck(enc_output)
        return enc_output, encoder_self_attn


class EncoderLayerGroup(nn.Module):
    def __init__(self, layer_num, d_features, n_head, d_k, d_v, dropout, d_bottleneck=128):
        super().__init__()
        self.inner_shared_multi_head_attn = MultiHeadAttention(n_head, d_features, d_k, d_v, dropout)
        self.inner_shared_position_wise_ff = PositionwiseFeedForward(d_features, d_bottleneck, dropout=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(self.inner_shared_multi_head_attn,
                                                       self.inner_shared_position_wise_ff) for _ in range(layer_num)])

    def forward(self, input_with_position, non_pad_mask=None, slf_attn_mask=None):
        encoder_self_attn_list = []
        for enc_layer in self.layer_stack:
            input_with_position, encoder_self_attn = enc_layer(input_with_position, non_pad_mask=non_pad_mask,
                                                               slf_attn_mask=slf_attn_mask)
            encoder_self_attn_list += [encoder_self_attn]
        output = input_with_position
        return output, encoder_self_attn_list


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.functional.softmax

    def forward(self, query, key, value, mask=None):
        '''
            Arguments:
                query {Tensor, shape [ batch, n_head, q_length, dk]} -- query
                key {Tensor, shape [ batch, n_head, k_length, dk]} -- key
                value {Tensor, shape [ batch, n_head, k_length, dv]} -- value
                mask {Tensor, shape [batch, 1, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [n_head * batch, q_length, dv] -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention

        '''
        attn = torch.matmul(query / self.temperature, key.transpose(2, 3))  # [batch, n_head, q_length, k_length]

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn, dim=-1)  # softmax over k_length
        attn = self.dropout(attn)  # [batch, n_head, q_length, k_length]
        output = torch.matmul(attn, value)  # [batch, n_head, q_length, dv]
        # output = value

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_features, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_features, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_features, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_features, n_head * d_v, bias=False)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_features + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_features + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_features + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_features)

        self.fc = nn.Linear(n_head * d_v, d_features, bias=False)
        # nn.init.xavier_normal_(self.fc.weight)

    def forward(self, query, key, value, mask=None):
        '''
            Arguments:
                query {Tensor, shape [batch, q_length, d_features]} -- query
                key {Tensor, shape [batch, k_length, d_features]} -- key
                value {Tensor, shape [batch, v_length, d_features]} -- value
                mask {Tensor, shape [batch, q_length, k_length]} --self attn mask

            Returns:
                output {Tensor, shape [batch, q_length, d_features]} -- output
                attn {Tensor, shape [n_head * batch, q_length, k_length] -- self attention
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = query.size()
        sz_b, len_k, _ = key.size()
        sz_b, len_v, _ = value.size()

        assert len_k == len_v

        residual = query

        query = self.layer_norm(query)

        query = self.w_qs(query).view(sz_b, len_q, n_head, d_k)  # target
        # [batch_size, seq_length, w2v_length] -> [batch_size, seq_length, n_head * dk] -> [batch_size, seq_length, n_head, dk]
        key = self.w_ks(key).view(sz_b, len_k, n_head, d_k)
        value = self.w_vs(value).view(sz_b, len_v, n_head, d_v)

        # query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [n_head * batch_size, seq_length, dk]
        # key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        # value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        query = query.transpose(1, 2).contiguous()  # [batch_size, n_head, seq_length, dq]
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if mask is not None and len(mask) != 0:
            # mask = mask.repeat(n_head, 1, 1)  # [n_head * batch_size, seq_length, seq_length]
            mask = mask.unsqueeze(1)
        output, attn = self.attention(query, key, value, mask=mask)  # [batch, n_head, q_length, dv]

        # output = output.view(sz_b, n_head, len_q, d_v)
        # output = output.permute(0, 2, 1, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.fc(output)
        output = self.dropout(output)
        output = output + residual

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            Arguments:
                x {Tensor, shape [batch_size, length, d_features]}

            Returns:
                x {Tensor, shape [batch_size, length, d_features]}

        '''

        residual = x
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = nn.functional.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x += residual
        return x


class SinusoidPositionEncoding(nn.Module):

    def __init__(self, d_features, max_length=None, d_meta=None):
        super().__init__()
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_length, d_features, padding_idx=0),
            freeze=True)

    def forward(self, x):
        '''
            Argument:
                x {Tensor, shape: [batch, length]} -- sequence position index masked
            Returns:
                x {Tensor, shape: [batch, length, d_features]} -- positional encoding
        '''
        x = self.position_enc(x)
        return x


class LinearPositionEncoding(nn.Module):
    def __init__(self, d_features, max_length=None, d_meta=None):
        super().__init__()
        self.position_enc = nn.Linear(d_meta, d_features, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.position_enc(x)
        x = self.tanh(x)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hid)
        self.activation = nn.functional.leaky_relu
        self.fc2 = nn.Linear(d_hid, d_out)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TimeFacilityEncoding(nn.Module):

    def __init__(self, d_features, max_length=None, d_meta=None):
        super().__init__()
        self.time_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_length, d_features, padding_idx=0),
            freeze=True)
        torch.manual_seed(1)
        self.facility_enc = torch.nn.Embedding(d_meta + 1, d_features, padding_idx=0)
        self.facility_enc.weight.requires_grad = False

    def forward(self, x):
        '''
            Argument:
                x {Tensor, shape: [batch, length, 2]} -- sequence position index masked
            Returns:
                x {Tensor, shape: [batch, length, d_features]} -- positional encoding
        '''
        facility_index = x[:, :, 1].long()
        facility_mask = facility_index == 0
        facility = self.facility_enc(facility_index)
        time = x[:, :, 0].masked_fill(facility_mask, 0)
        time = self.time_enc(time)

        x = time + facility
        return x
