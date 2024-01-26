from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron model.

    It uses leaky ReLU as activation function and the last layer is a
    linear layer.

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    hidden_dim : int
        Dimension of the hidden layers.
    output_dim : int
        Dimension of the output.
    n_layers : int
        Number of hidden layers.
    use_softmax : bool
        Whether to use softmax as the activation function of the last layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        use_softmax: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.use_softmax = use_softmax
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.functional.leaky_relu(x)
        x = self.layers[-1](x)
        if self.use_softmax:
            x = nn.functional.softmax(x, dim=1)
        x = x.squeeze(0)
        return x


class CNN1D(nn.Module):
    """1D convolutional neural network model.

    It uses leaky ReLU as activation function and the last layer is a
    linear layer.

    Parameters
    ----------
    input_channel : int
        Number of channels in the input.
    ncs : list of int.
        Number of channels in each convolution layer.
    nds : list of int.
        Number of neurons in each dense layer.
    nf : int
        Input length of the convolution layer.
    stride : int
        Stride of the convolution.
    kernel_size : int
        Size of the kernel.
    """

    def __init__(self, input_channel, ncs, nds, nf=1024, stride=2, kernel_size=7):
        super(CNN1D, self).__init__()
        self.input_channel = input_channel
        self.ncs = ncs
        self.nds = nds
        self.stride = stride
        self.activation = nn.LeakyReLU()

        nc_old = input_channel
        self.kernel_size = kernel_size
        self.conv = nn.ModuleList()
        self.nf = nf

        for i in range(len(ncs)):
            self.conv.append(
                nn.Conv1d(
                    in_channels=nc_old,
                    out_channels=ncs[i],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                )
            )
            # The if statement and the addition is necessary to guarantee that, when flattening the output of the last convolution layer,
            # we will have the same dimension as the first layer of the FCN
            if nf % 2 != 0:
                nf = nf // 2
                nf += 1
            else:
                nf = nf // 2
            nc_old = ncs[i]

        # FCN to send input to latent spaceha
        self.FCN = nn.ModuleList()
        nds_old = ncs[-1] * nf

        for i in range(len(nds)):
            self.FCN.append(nn.Linear(nds_old, nds[i]))
            nds_old = nds[i]

    def forward(self, x):
        resqueeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            resqueeze = True
        # do convolution
 
        for module in self.conv:
            x = module(x)
            x = self.activation(x)

        # flatten
        x = x.view(x.size(0), -1)
        # do FCN
        n_fcn = len(self.FCN)
        for i, module in enumerate(self.FCN):
            x = module(x)
            if i < n_fcn - 1:
                x = self.activation(x) 
        if resqueeze:
            x = x.squeeze(0)
        # if classification add sigmoid
        return x
