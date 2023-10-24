config_dict = dict()
config_dict['batch_size'] = 12
config_dict['seed'] = 42
config_dict['lr'] = 1e-4
config_dict['beta_1'] = 0.9
config_dict['beta_2'] = 0.98
config_dict['epochs'] = 100
config_dict['n_hidden_dim'] = 64
config_dict['checkpoint_path'] = './train_logs/lightning_logs/version_14/checkpoints/epoch=75-step=57000.ckpt'


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict(config_dict)
