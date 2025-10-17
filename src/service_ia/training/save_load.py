import logging
import os

import joblib
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.DEBUG)


class SaveLoad:
    """
    Classe per salvare e caricare modelli sklearn
    """

    def __init__(self, **kwargs):
        self.save_pkl = kwargs.get('save_pkl', False)
        self.filename = kwargs.get('filename', 'best_model')
        self.generate_filename()

    def generate_filename(self):
        """
        Genera il percorso completo del file per salvare il modello
        :return: percorso completo del file
        """
        self.filename = os.path.abspath(
            os.path.join('best_models', self.filename if self.filename.endswith('.pkl') else f'{self.filename}.pkl'))

    def save_model(self, estimator):
        """
        Salva il modello addestrato su file
        :param estimator: modello sklearn addestrato
        """
        if self.save_pkl:
            joblib.dump(estimator, self.filename)
            logging.info(f'Modello salvato in {self.filename}')

    def load_model(self):
        """
        :return: modello sklearn caricato da file
        """
        if os.path.exists(self.filename):
            estimator = joblib.load(self.filename)
            logging.info(f'Modello caricato da {self.filename}')
            return estimator
        else:
            logging.error(f'File modello non trovato: {self.filename}')
            return None
