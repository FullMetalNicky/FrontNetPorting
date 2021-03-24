from PreActBlock import PreActBlock
from ConvBlock import ConvBlock
from FrontNet import FrontNet
from Dronet import Dronet
from PenguiNet import PenguiNet

from torchsummary import summary


"""This is the model referred to in the paper as FrontNet

      possible configurations are:
        input size 160x96 (w,h) - c=32, fc_nodes=1920 (default)
        input size 160x96 (w,h) - c=16, fc_nodes=960
        input size 80x48 (w,h) - c=32, fc_nodes=768
      Where c is the number of the channels in the first convolution layer
      The model in the example is configured to handle gray-scale input 
"""


def PenguiNetModel(h=96, w=160, c=32, fc_nodes=1920):

    model = PenguiNet(ConvBlock, [1, 1, 1], gray=True, c=c, fc_nodes=fc_nodes)
    summary(model, (1, h, w))

    return model


"""This model is the Dronet architecture (TF) implemented in PyTorch
    It works with input size 160x96 (w,h), with 32 channels in the first convolution layer
    The model in the example is configured to handle gray-scale input 
"""

def DronetModel():

    model = Dronet(PreActBlock, [1, 1, 1], gray=True)
    summary(model, (1, 96, 160))

    return model


"""This model is the ProxmityNet of Dario (Keras) implemented in PyTorch
    It works with input size 160x96 (w,h), with 32 channels in the first convolution layer
    The model in the example is configured to handle RGB input 
"""

def FrontNetModel():

    model = FrontNet(PreActBlock, [1, 1, 1], gray=False)
    summary(model, (3, 60, 108))

    return model
