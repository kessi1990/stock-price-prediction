import torch
import torch.nn as nn
import torch.nn.functional as functional


class StockPredictor(nn.Module):
    """
    model class for stock predictor. model consists of an LSTM cell and dense / fully connected layer
    """
    def __init__(self, input_size, hidden_size, num_layers, device):
        """
        init method / constructor
        :param input_size: number of input features
        :param hidden_size: number of features of the hidden state and cell state
        :param num_layers: number of subsequent layers of the LSTM cell
        :param device: device that is used for training (cpu / gpu)
        """
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        """
        forwards input data through the model
        :param x: tensor containing the input data
        :return: returns the prediction
        """
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device, requires_grad=True)
        c_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=self.device, requires_grad=True)
        out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class StockPredictorAttention(nn.Module):
    """
    model class for stock predictor, integrates the attention mechanism.
    model consists of an LSTM cell, attention layer and a final dense / fully connected layer
    """
    def __init__(self, input_size, hidden_size, num_layers, device, alignment='concat'):
        """
        init method / constructor
        :param input_size: number of input features
        :param hidden_size: number of features of the hidden state and cell state
        :param num_layers: number of subsequent layers of the LSTM cell
        :param device: device that is used for training (cpu / gpu)
        :param alignment: specifies the alignment method which is used in the additional attention layer
        """
        super(StockPredictorAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.alignment = alignment

        self.h_0 = None
        self.c_0 = None

        self.init_hidden()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.attention = Attention(alignment=alignment, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        """
        forwards input data through the model
        :param x: tensor containing the input data
        :return: returns the prediction
        """
        self.init_hidden(x.size(0))
        x = x.permute((1, 0, 2)).unsqueeze(dim=2)
        out = None
        for input_seq in x:
            out, (self.h_0, self.c_0) = self.lstm(input_seq, (self.h_0.detach(), self.c_0.detach()))
            out = self.attention(out, self.h_0[-1])
            out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size=1):
        """
        initializes the hidden state and cell state of the LSTM with zeros
        :param batch_size: defines the sizes of the batch, default = 1
        :return: None
        """
        self.h_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device, requires_grad=True)
        self.c_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device, requires_grad=True)


class Attention(nn.Module):
    """
    attention layer, applies attention mechanism
    """
    def __init__(self, alignment, hidden_size):
        """
        init method / constructor
        :param alignment: alignment method for computing the alignment score
        :param hidden_size: number of features of LSTM's hidden state and cell state -> feature size of attention layer
        """
        super(Attention, self).__init__()
        self.alignment = alignment
        bias = True
        if self.alignment == 'general':
            self.fc_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)
        elif self.alignment == 'concat':
            self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=bias)
        elif self.alignment == 'concat_fc':
            self.fc_1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=bias)
            self.fc_3 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)
        else:  # self.alignment == 'dot'
            pass
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=1, bias=bias)

    def forward(self, input_vectors, last_hidden_state):
        """
        compute attention based on all input vectors and last hidden_state of LSTM
        :param input_vectors: vectorized input data
        :param last_hidden_state: last hidden state of LSTM
        :return: context vector z
        """
        # b = batch
        # s = sequence length
        # f = features
        # last_hidden_state (b, f) -> unsqueeze(dim=1) -> (b, 1, 128)
        if self.alignment == 'general':
            """
            # general
            # align(v_it, h_t−1) = h^T_t−1 * (W_a(v_it) + b_a)
            # --------------------------------------------------------------------------------------------------------
            # 1. weights matrix with bias (fc_1) -> (b, s, f) 
            # 2. dot product transposed last_hidden_state (b, 1, f)^T * input_vectors (b, s, f)
            # --------------------------------------------------------------------------------------------------------
            """
            alignment_scores = torch.bmm(self.fc_1(input_vectors), last_hidden_state.unsqueeze(dim=1).permute(0, 2, 1))
        elif self.alignment == 'concat':
            """
            # concat
            # align(v_it, h_t−1) = W_s(tanh(W_a[v_it ; h_t−1] + b_a)) + b_s
            # --------------------------------------------------------------------------------------------------------
            # 1. concat input_vectors (b, s, f) and last_hidden_state (b, 1, f) -> (b, s, 2*f)
            # 2. weights matrix with bias (fc_1) -> (b, s, f) 
            # 3. apply hyperbolic tangent function -> aligned input_vectors (b, s, f)
            # 4. alignment_score for each input_vector regarding last_hidden_state:
            # -> aligned input_vectors (b, s, f) -> weights matrix with bias (fc_2) -> alignment_scores (b, s, 1)
            # --------------------------------------------------------------------------------------------------------
            """
            # batch, seq_len, features
            _, seq_len, _ = input_vectors.shape
            alignment_scores = self.fc_2(torch.tanh(self.fc_1(
                torch.cat((input_vectors, last_hidden_state.unsqueeze(dim=1).expand(-1, seq_len, -1)), dim=-1))))
        elif self.alignment == 'concat_fc':
            """
            # concat_fc
            # align(v_it, h_t−1) = W_s(tanh(W_a[v_it ; W_h(h_t−1) + b_h] + b_a)) + b_s
            # --------------------------------------------------------------------------------------------------------
            # 1. weights matrix with bias (fc_3) to last_hidden_state -> (b, 1, f) 
            # 2. concat input_vectors (b, s, f) and last_hidden_state (b, 1, f) -> (b, s, 2*f)
            # 3. weights matrix with bias (fc_1) -> (b, s, f) 
            # 4. apply hyperbolic tangent function -> aligned input_vectors (b, s, f)
            # 5. alignment_score for each input_vector regarding last_hidden_state:
            # -> aligned input_vectors (b, s, f) -> weights matrix with bias (fc_2) -> alignment_scores (b, s, 1)
            # --------------------------------------------------------------------------------------------------------
            """
            # batch, seq_len, features
            _, seq_len, _ = input_vectors.shape
            alignment_scores = self.fc_2(torch.tanh(self.fc_1(
                torch.cat((input_vectors, self.fc_3(last_hidden_state).unsqueeze(dim=1).expand(-1, seq_len, -1)),
                          dim=-1))))
        else:
            """
            # dot
            # align(v_it, h_t−1) = h^T_t−1 * v_it
            # --------------------------------------------------------------------------------------------------------
            # 1. dot product transposed last_hidden_state (b, 1, f)^T * input_vectors (b, s, f)
            # --------------------------------------------------------------------------------------------------------
            """
            alignment_scores = torch.bmm(input_vectors, last_hidden_state.unsqueeze(dim=1).permute(0, 2, 1))
        """
        # softmax + linear combination
        # --------------------------------------------------------------------------------------------------------
        # apply softmax function to dim=1 -> importance of each input_vector -> attention_weights (b, s, 1)
        # pointwise multiplication of input_vectors (b, s, f) and their corresponding attention value (b, s, 1)  -> (b, s, f)
        # compute sum of these products (b, s, f) along dim=1 to obtain context_vector z (b, 1, f)  |  == linear combination
        # --------------------------------------------------------------------------------------------------------
        """
        attention_weights = functional.softmax(alignment_scores, dim=1)
        context = input_vectors * attention_weights
        z = torch.sum(context, dim=1, keepdim=True)
        """
        z = torch.bmm(attention_weights.permute(0, 2, 1), input_vectors)
        """
        return z
