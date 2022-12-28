import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as e2nn


class DSteerableLeNet(nn.Module):
    """
    Steerable CNN for image classification.

    Functions:
        forward: forward pass of the network
    """

    def __init__(self, imsize=151, kernel_size=5, N=16):
        """
        Initialize the network.

        Args:
            imsize: image size
            kernel_size: kernel size of the convolutional layers
            N: number of filters in the convolutional layers
        """
        super(DSteerableLeNet, self).__init__()
        self.imsize = imsize
        self.kernel_size = kernel_size
        self.N = N

        z = 0.5 * (self.imsize - 2)
        z = int(0.5 * (z - 2))

        self.r2_act = gspaces.FlipRot2dOnR2(self.N)

        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type

        out_type = e2nn.FieldType(self.r2_act, 6 * [self.r2_act.regular_repr])
        self.mask = e2nn.MaskModule(in_type, self.imsize, margin=1)
        self.conv1 = e2nn.R2Conv(
            in_type, out_type, kernel_size=self.kernel_size, padding=1, bias=False
        )
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        in_type = self.pool1.out_type
        out_type = e2nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(
            in_type, out_type, kernel_size=self.kernel_size, padding=1, bias=False
        )
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPoolAntialiased(out_type, kernel_size=2)

        self.gpool = e2nn.GroupPooling(out_type)

        self.fc = nn.Linear(16 * z * z, 2048)

        # dummy parameter for tracking device
        self.dummy = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x: input data
        """
        x = e2nn.GeometricTensor(x, self.input_type)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.gpool(x)
        x = x.tensor

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x
