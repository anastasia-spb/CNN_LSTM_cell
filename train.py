import os
from typing import Any

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from MovingMNIST.MovingMNIST import MovingMNIST
import pytorch_lightning as pl
from model import EncoderDecoderConvLSTM
from torchvision import transforms
from PIL import Image

from config import config as config


class FrameTransform(object):
    def __call__(self, frame: Image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        return transform(frame).to(torch.float32)


class MovingMNISTLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingMNISTLightning, self).__init__()

        # default config
        self.normalize = False
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = config.batch_size
        self.n_steps_past = 10
        self.n_steps_ahead = 10

    def forward(self, x):
        # add channel
        x = x[:, :, None, :, :]
        output = self.model(x, future_seq=self.n_steps_ahead)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.squeeze(y_hat, dim=1)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.squeeze(y_hat, dim=1)
        loss = self.criterion(y_hat, y)

        self.log("test_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None, results_folder: str = './predict_result') -> Any:
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.squeeze(y_hat, dim=1)
        y_hat = torch.sigmoid(y_hat)

        y_hat = y_hat.detach().cpu().numpy()
        for idx_in_batch, frame_seq in enumerate(y_hat):
            for idx_in_frame, frame in enumerate(frame_seq):
                predicted_mask = Image.fromarray((frame * 255).astype('uint8'))
                frame_tag = ''.join((str(batch_idx), '_', str(idx_in_batch), '_', str(idx_in_frame), '.png'))
                predicted_mask.save(os.path.join(results_folder, frame_tag))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.lr, betas=(config.beta_1, config.beta_2))


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

    transform = FrameTransform()

    data_path = os.path.join(os.getcwd(), 'moving_mnist_data')
    test_data = MovingMNIST(root=data_path, train=False, download=False, transform=transform,
                            target_transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        shuffle=True)

    train_data = MovingMNIST(root=data_path, train=True, download=False, transform=transform,
                             target_transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True)

    conv_lstm_model = EncoderDecoderConvLSTM(nf=config.n_hidden_dim, in_chan=1)
    model = MovingMNISTLightning(model=conv_lstm_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        monitor="test_loss",
        mode="min",
        save_top_k=1
    )

    trainer = pl.Trainer(default_root_dir="train_logs",
                         max_epochs=config.epochs,
                         enable_checkpointing=True,
                         accelerator=accelerator,
                         callbacks=[checkpoint_callback],
                         )

    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    train()
