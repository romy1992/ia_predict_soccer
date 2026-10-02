import logging
import os

from src.storage import bucket_store

logging.basicConfig(level=logging.DEBUG)


class SaveLoad:
    """
    Salva/carica modelli sklearn sul Bucket S3-compatible (non piu'
    filesystem locale): `scheduler`, che esegue il retrain automatico
    giornaliero, non ha mai avuto un Volume montato - ogni scrittura
    locale andava persa ad ogni redeploy e non era mai visibile ad `api`.
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
        Genera la CHIAVE bucket per il modello - sempre '/' (mai
        `os.path.join`, che su Windows userebbe '\\' e produrrebbe una
        chiave S3 non valida).

        `filename` puo' contenere una sottocartella relativa a `best_models`
        (es. `under_over/under_over_1_5/under_over_1_5_champion_20260914`):
        i mercati rifatti con la procedura nuova salvano direttamente nella
        propria "cartella" (prefisso di chiave).
        :return: chiave bucket completa
        """
        name = self.filename if self.filename.endswith('.pkl') else f'{self.filename}.pkl'
        name = name.replace('\\', '/').lstrip('/')
        self.key = f'best_models/{name}'
        self.filename = self.key

    def save_model(self, estimator, **metadata):
        """
        Salva il modello addestrato sul bucket.
        :param estimator: modello sklearn addestrato
        """
        if self.save_pkl:
            bucket_store.put_joblib(self.key, estimator)
            logging.info(f'Modello salvato sul bucket: {self.key}')

            if self.registry_enabled:
                from src.service_ia.training.model_registry import ModelRegistry

                registry = ModelRegistry(registry_dir=self.registry_dir)
                registry.register(
                    model_path=self.key,
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
        :return: modello sklearn caricato dal bucket, o None se assente
        """
        if bucket_store.exists(self.key):
            estimator = bucket_store.get_joblib(self.key)
            logging.info(f'Modello caricato dal bucket: {self.key}')
            return estimator
        else:
            logging.error(f'File modello non trovato sul bucket: {self.key}')
            return None
