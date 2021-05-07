
from HimaxDataModule import HimaxDataModule
from PenguiNet import PenguiNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


data_module = HimaxDataModule("Data/160x96OthersTrainsetAug.pickle", "Data/160x96StrangersTestset.pickle")
model = PenguiNet()

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=40,
   verbose=False,
   mode='max'
)
checkpoint_callback = ModelCheckpoint(
   monitor='val_loss',
   filename='PenguiNet{epoch}',
   save_top_k=4
)

trainer = pl.Trainer(min_epochs=2, max_epochs=100, gpus=1, callbacks=[checkpoint_callback])
model = PenguiNet.load_from_checkpoint("lightning_logs/version_0/checkpoints/PenguiNetepoch=88.ckpt")
trainer.fit(model, data_module)
#data_module.setup()
#trainer.test(model, datamodule=data_module)