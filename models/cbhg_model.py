"""
The CBHG model implementation
"""

from typing import List, Optional, Any

from torch import nn
import torch


class BatchNormConv1d(nn.Module):
    """
    A nn.Conv1d followed by an optional activation function, and nn.BatchNorm1d
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: Any = None,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

    def forward(self, x: Any):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    """Highway Networks were developed by (Srivastava et al., 2015)
    to overcome the difficulty of training deep neural networks
    (https://arxiv.org/abs/1507.06228).
    Args:
    in_size (int): the input size
    out_size (int): the output size
    """

    def __init__(self, in_size, out_size):
        """
        Initializing Highway networks
        """
        super().__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor):
        """Calculate forward propagation
        Args:
        inputs (Tensor):
        """
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Prenet(nn.Module):
    """
    A prenet is a collection of linear layers with dropout(0.5),
    and RELU activation function
    Args:
    config: the hyperparameters object
    in_dim (int): the input dim
    """

    def __init__(self, in_dim: int, prenet_depth: List[int] = [256, 128], dropout: int = 0.5):
        """Initializing the prenet module"""
        super().__init__()
        in_sizes = [in_dim] + prenet_depth[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size) in zip(in_sizes, prenet_depth)]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        """Calculate forward propagation
        Args:
        inputs (batch_size, seqLen): the inputs to the prenet, the input shapes could
        be different as it is being used in both encoder and decoder.
        Returns:
        Tensor: the output of  the forward propagation
        """
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class CBHG(nn.Module):
    """The CBHG module (1-D Convolution Bank + Highway network + Bidirectional GRU)
    was proposed by (Lee et al., 2017, https://www.aclweb.org/anthology/Q17-1026)
    for a character-level NMT model.
    It was adapted by (Wang et al., 2017) for building the Tacotron.
    It is used in both the encoder and decoder  with different parameters.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int,
        projections: List[int],
        highway_layers: int = 4,
    ):
        """Initializing the CBHG module
        Args:
        in_dim (int): the input size
        out_dim (int): the output size
        k (int): number of filters
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [
                BatchNormConv1d(
                    in_dim,
                    in_dim,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    activation=self.relu,
                )
                for k in range(1, K + 1)
            ]
        )
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [
                BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1, padding=1, activation=ac)
                for (in_size, out_size, ac) in zip(in_sizes, projections, activations)
            ]
        )

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList([Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(in_dim, out_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs
        x = x.transpose(1, 2)
        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class CBHGModel(nn.Module):
    """CBHG model implementation as described in the paper:
     https://ieeexplore.ieee.org/document/9274427

    Args:
    inp_vocab_size (int): the number of the input symbols
    targ_vocab_size (int): the number of the target symbols (diacritics)
    embedding_dim (int): the embedding  size
    use_prenet (bool): whether to use prenet or not
    prenet_sizes (List[int]): the sizes of the prenet networks
    cbhg_gru_units (int): the number of units of the CBHG GRU, which is the last
    layer of the CBHG Model.
    cbhg_filters (int): number of filters used in the CBHG module
    cbhg_projections: projections used in the CBHG module

    Returns:
    diacritics Dict[str, Tensor]:
    """

    def __init__(
        self,
        inp_vocab_size: int,
        targ_vocab_size: int,
        embedding_dim: int = 512,
        use_prenet: bool = True,
        prenet_sizes: List[int] = [512, 256],
        cbhg_gru_units: int = 512,
        cbhg_filters: int = 16,
        cbhg_projections: List[int] = [128, 256],
        post_cbhg_layers_units: List[int] = [256, 256],
        post_cbhg_use_batch_norm: bool = True,
    ):
        super().__init__()
        self.use_prenet = use_prenet
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)
        if self.use_prenet:
            self.prenet = Prenet(embedding_dim, prenet_depth=prenet_sizes)

        self.cbhg = CBHG(
            prenet_sizes[-1] if self.use_prenet else embedding_dim,
            cbhg_gru_units,
            K=cbhg_filters,
            projections=cbhg_projections,
        )

        layers = []
        post_cbhg_layers_units = [cbhg_gru_units] + post_cbhg_layers_units

        for i in range(1, len(post_cbhg_layers_units)):
            layers.append(
                nn.LSTM(
                    post_cbhg_layers_units[i - 1] * 2,
                    post_cbhg_layers_units[i],
                    bidirectional=True,
                    batch_first=True,
                )
            )
            if post_cbhg_use_batch_norm:
                layers.append(nn.BatchNorm1d(post_cbhg_layers_units[i] * 2))

        self.post_cbhg_layers = nn.ModuleList(layers)
        self.projections = nn.Linear(post_cbhg_layers_units[-1] * 2, targ_vocab_size)
        self.post_cbhg_layers_units = post_cbhg_layers_units
        self.post_cbhg_use_batch_norm = post_cbhg_use_batch_norm

    def forward(
        self,
        src: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ):
        """Compute forward propagation"""

        # src = [batch_size, src len]
        # lengths = [batch_size]
        # target = [batch_size, trg len]

        embedding_out = self.embedding(src)
        # embedding_out; [batch_size, src_len, embedding_dim]

        cbhg_input = embedding_out
        if self.use_prenet:
            cbhg_input = self.prenet(embedding_out)

            # cbhg_input = [batch_size, src_len, prenet_sizes[-1]]

        outputs = self.cbhg(cbhg_input, lengths)

        hn = torch.zeros((2, 2, 2))
        cn = torch.zeros((2, 2, 2))

        for i, layer in enumerate(self.post_cbhg_layers):
            if isinstance(layer, nn.BatchNorm1d):
                outputs = layer(outputs.permute(0, 2, 1))
                outputs = outputs.permute(0, 2, 1)
                continue
            if i > 0:
                outputs, (hn, cn) = layer(outputs, (hn, cn))
            else:
                outputs, (hn, cn) = layer(outputs)

        predictions = self.projections(outputs)

        # predictions = [batch_size, src len, targ_vocab_size]

        output = {"diacritics": predictions}

        return output
