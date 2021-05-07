import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from ConvBlock import ConvBlock
from torch.nn import functional as F

class PenguiNet(LightningModule):

  def __init__(self):
    super().__init__()

    self.name = "PenguiNet"
    self.inplanes = 32
    self.width = 160
    self.height = 96
    self.dilation = 1
    self._norm_layer = nn.BatchNorm2d

    self.groups = 1
    self.base_width = 64
    self.conv = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.bn = nn.BatchNorm2d(self.inplanes)
    self.relu1 = nn.ReLU()

    self.layer1 = ConvBlock(self.inplanes, self.inplanes, stride=2)
    self.layer2 = ConvBlock(self.inplanes, self.inplanes * 2, stride=2)
    self.layer3 = ConvBlock(self.inplanes * 2, self.inplanes * 4, stride=2)

    self.dropout = nn.Dropout()
    self.fc = nn.Linear(1920, 4)

  def forward(self, x):
    batch_size, channels, width, height = x.size()
    conv5x5 = self.conv(x)
    btn = self.bn(conv5x5)
    relu1 = self.relu1(btn)
    max_pool = self.maxpool(relu1)

    l1 = self.layer1(max_pool)
    l2 = self.layer2(l1)
    l3 = self.layer3(l2)
    out = l3.flatten(1)

    out = self.dropout(out)
    out = self.fc(out)
    x = out[:, 0]
    y = out[:, 1]
    z = out[:, 2]
    phi = out[:, 3]
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)
    phi = phi.unsqueeze(1)

    return [x, y, z, phi]

  def training_step(self, batch, batch_idx):
    x, y = batch
    outputs = self.forward(x)
    loss_x = F.l1_loss(outputs[0], (y[:, 0]).view(-1, 1))
    loss_y = F.l1_loss(outputs[1], (y[:, 1]).view(-1, 1))
    loss_z = F.l1_loss(outputs[2], (y[:, 2]).view(-1, 1))
    loss_phi = F.l1_loss(outputs[3], (y[:, 3]).view(-1, 1))
    loss = loss_x + loss_y + loss_z + loss_phi
    self.log('train_loss', loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    outputs = self.forward(x)
    loss_x = F.l1_loss(outputs[0], (y[:, 0]).view(-1, 1))
    loss_y = F.l1_loss(outputs[1], (y[:, 1]).view(-1, 1))
    loss_z = F.l1_loss(outputs[2], (y[:, 2]).view(-1, 1))
    loss_phi = F.l1_loss(outputs[3], (y[:, 3]).view(-1, 1))
    loss = loss_x + loss_y + loss_z + loss_phi
    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    return optimizer

  def test_step(self, batch, batch_idx):
    x, y = batch
    outputs = self.forward(x)
    loss_x = F.l1_loss(outputs[0], (y[:, 0]).view(-1, 1))
    loss_y = F.l1_loss(outputs[1], (y[:, 1]).view(-1, 1))
    loss_z = F.l1_loss(outputs[2], (y[:, 2]).view(-1, 1))
    loss_phi = F.l1_loss(outputs[3], (y[:, 3]).view(-1, 1))
    loss = loss_x + loss_y + loss_z + loss_phi
    self.log('test_loss', loss)
    return loss