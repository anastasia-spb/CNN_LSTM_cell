import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from MovingMNIST.MovingMNIST import MovingMNIST
import pytorch_lightning as pl
from model import EncoderDecoderConvLSTM
from torchvision import transforms
from PIL import Image
from collections import OrderedDict

from config import config as config
from train import FrameTransform, MovingMNISTLightning


def get_state_dict(loaded_dict):
    state_dict = OrderedDict()
    for k, v in loaded_dict.items():
        name = k.replace("model.", '')
        state_dict[name] = v
    return state_dict


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

    transform = FrameTransform()

    data_path = os.path.join(os.getcwd(), 'moving_mnist_data')
    test_data = MovingMNIST(root=data_path, train=False, download=False, transform=transform,
                            target_transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        shuffle=True)

    checkpoint = torch.load(config.checkpoint_path)
    conv_lstm_model = EncoderDecoderConvLSTM(nf=config.n_hidden_dim, in_chan=1)
    state_dict = get_state_dict(checkpoint['state_dict'])
    conv_lstm_model.load_state_dict(state_dict, strict=True)

    model = MovingMNISTLightning(model=conv_lstm_model)

    trainer = pl.Trainer(accelerator=accelerator)
    trainer.predict(model, test_loader)


if __name__ == '__main__':
    predict()
