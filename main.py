import os
from typing import Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from data.mm import MovingMNIST
from conv_lstm_model import create_model
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image

from config import config as config


def prepare_data(data_dir: str, batch_size: int = 12, frames_input: int = 10, frames_output: int = 10):
    trainFolder = MovingMNIST(is_train=True,
                          root=data_dir,
                          n_frames_input=frames_input,
                          n_frames_output=frames_output,
                          num_objects=[3])
    validFolder = MovingMNIST(is_train=False,
                          root=data_dir,
                          n_frames_input=frames_input,
                          n_frames_output=frames_output,
                          num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=config.num_workers)
    validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=config.num_workers)
    return trainLoader, validLoader


class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None, predict_folder: str=''):
        super(MovingMNISTLightning, self).__init__()
        self.save_hyperparameters()

        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.predict_folder = predict_folder


    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        idx, targetVar, inputVar, _, _ = batch # B,S,C,H,W
        pred = self.forward(inputVar)
        loss = self.criterion(pred, targetVar)

        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targetVar, inputVar, _, _ = batch # B,S,C,H,W
        pred = self.forward(inputVar)
        loss = self.criterion(pred, targetVar)

        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None) -> Any:
        _, _, inputVar, _, _ = batch # B,S,C,H,W
        pred = self.forward(inputVar)
        pred = torch.squeeze(pred, dim=2)
        pred_softmaxes = torch.sigmoid(pred).detach().cpu().numpy()
        for idx_in_batch, frame_seq in enumerate(pred_softmaxes):
            for idx_in_frame, frame in enumerate(frame_seq):
                predicted_mask = Image.fromarray((frame * 255).astype('uint8'))
                frame_tag = ''.join((str(batch_idx), '_', str(idx_in_batch), '_', str(idx_in_frame), '.png'))
                predicted_mask.save(os.path.join(self.predict_folder, frame_tag))


    def configure_optimizers(self):
        optimizer_config = {
            'params' : self.model.parameters(),
            'lr' : self.hparams.lr,
            'weight_decay' : self.hparams.weight_decay
        }
        
        optimizer = torch.optim.Adam(**optimizer_config)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  factor=0.5,
                                                                  patience=4,
                                                                  verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': lr_scheduler,
           'monitor': 'val_loss'
       }




def train():
    torch.random.manual_seed(config.seed)
    pl.seed_everything(config.seed)

    accelerator = 'cpu'
    if torch.cuda.is_available():
        # Cuda maintenance
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(config.seed)
        accelerator = 'gpu'

    print(f"Accelerator: {accelerator}")

    train_loader, valid_loader = prepare_data(data_dir=config.data_dir)

    conv_lstm_model = create_model()
    model = MovingMNISTLightning(model=conv_lstm_model, predict_folder=config.predict_folder)

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        monitor="test_loss",
        mode="min",
        save_top_k=1
    )

    trainer_kwargs = dict()
    trainer_kwargs["accelerator"] = accelerator
    trainer_kwargs["devices"] = config.devices
    if (accelerator == 'gpu') and (len(config.devices) > 1):
        trainer_kwargs["strategy"] = 'dp' # 'ddp'


    trainer = pl.Trainer(default_root_dir="train_logs",
                         max_epochs=config.epochs,
                         enable_checkpointing=True,
                         accelerator=accelerator,
                         callbacks=[checkpoint_callback],
                         **trainer_kwargs
                         )

    trainer.fit(model, train_loader, valid_loader)


def predict():
    torch.random.manual_seed(config.seed)
    pl.seed_everything(config.seed)

    accelerator = 'cpu'
    if torch.cuda.is_available():
        # Cuda maintenance
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(config.seed)
        accelerator = 'gpu'

    print(f"Accelerator: {accelerator}")

    _, valid_loader = prepare_data(data_dir=config.data_dir)

    conv_lstm_model = create_model()
    model_info = torch.load(config.checkpoint_path)
    conv_lstm_model.load_state_dict(model_info['state_dict'], strict=True)
    model = MovingMNISTLightning(model=conv_lstm_model, predict_folder=config.predict_folder)

    trainer = pl.Trainer(accelerator=accelerator, devices=[0])
    trainer.predict(model, valid_loader)



if __name__ == '__main__':
    # train()
    predict()