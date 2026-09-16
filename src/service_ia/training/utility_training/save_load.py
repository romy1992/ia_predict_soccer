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
        self.market_name = kwargs.get('market_name', 'generic')
        self.metrics = kwargs.get('metrics')
        self.feature_names = kwargs.get('feature_names')
        self.registry_enabled = kwargs.get('registry_enabled', True)
        self.registry_dir = kwargs.get('registry_dir', os.path.join('best_models', 'registry'))
        self.generate_filename()

    def generate_filename(self):
        """
        Genera il percorso completo del file per salvare il modello.

        `filename` puo' contenere una sottocartella relativa a `best_models`
        (es. `under_over/under_over_1_5/under_over_1_5_champion_20260914`):
        `save_model` crea la cartella se non esiste, quindi i mercati rifatti
        con la procedura nuova salvano direttamente nella propria cartella.
        :return: percorso completo del file
        """
        self.filename = os.path.abspath(
            os.path.join('best_models', self.filename if self.filename.endswith('.pkl') else f'{self.filename}.pkl'))

    def save_model(self, estimator, **metadata):
        """
        Salva il modello addestrato su file
        :param estimator: modello sklearn addestrato
        """
        if self.save_pkl:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            joblib.dump(estimator, self.filename)
            logging.info(f'Modello salvato in {self.filename}')

            if self.registry_enabled:
                from src.service_ia.training.model_registry import ModelRegistry

                registry = ModelRegistry(registry_dir=self.registry_dir)
                registry.register(
                    model_path=self.filename,
                    market=metadata.get('market_name', self.market_name),
                    model_name=metadata.get('model_name', estimator.__class__.__name__),
                    metrics=metadata.get('metrics', self.metrics),
                    feature_names=metadata.get('feature_names', self.feature_names),
                    params=metadata.get('params'),
                    extra=metadata.get('extra'),
                    dataset_version=metadata.get('dataset_version'),
                    feature_version=metadata.get('feature_version'),
                    windows=metadata.get('windows'),
                    git_sha=metadata.get('git_sha'),
                    stage=metadata.get('stage') or 'candidate',
                )

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
