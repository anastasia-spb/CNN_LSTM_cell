config_dict = dict()
config_dict['data_dir'] = '/workspaces/houghnet/MovingMNIST/data'
config_dict['batch_size'] = 12
config_dict['seed'] = 42
config_dict['lr'] = 1e-4
config_dict['epochs'] = 100
config_dict['devices'] = [0] #[0, 1, 2, 3]
config_dict["num_workers"] = 8
config_dict['checkpoint_path'] = '/workspaces/houghnet/save_model/2023-10-24T00-00-00/checkpoint_17_0.004963.pth.tar'
config_dict['predict_folder'] = '/workspaces/houghnet/MovingMNIST/data/predictions'


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict(config_dict)