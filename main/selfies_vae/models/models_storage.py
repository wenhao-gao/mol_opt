# from models.smiles_vae import VAE, VAETrainer, vae_parser
# from models.jtvae import JTVAE, JTVAETrainer, jtvae_parser
from models.selfies_vae import VAE, VAETrainer, vae_parser


class ModelsStorage():

    def __init__(self):
        self._models = {}
        # self.add_model('smiles_vae', VAE, VAETrainer, vae_parser)
        # self.add_model('jtvae', JTVAE, JTVAETrainer, jtvae_parser)
        self.add_model('selfies_vae', VAE, VAETrainer, vae_parser)

    def add_model(self, name, class_, trainer_, parser_):
        self._models[name] = {'class': class_,
                              'trainer': trainer_,
                              'parser': parser_}

    def get_model_names(self):
        return list(self._models.keys())

    def get_model_trainer(self, name):
        return self._models[name]['trainer']

    def get_model_class(self, name):
        return self._models[name]['class']

    def get_model_train_parser(self, name):
        return self._models[name]['parser']

