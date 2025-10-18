# src/config/config_drive.py
import io
import json
import os
import pickle
from pathlib import Path
from typing import Optional

from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload, MediaIoBaseDownload
from sklearn.ensemble import RandomForestClassifier

SCOPES = ['https://www.googleapis.com/auth/drive']
_service = None  # lazy singleton


def _load_credentials_by_workspace() -> Credentials:
    """
    Ordine di risoluzione:
    1) Env var GDRIVE_SA_JSON con il contenuto JSON delle credenziali
    2) Env var GOOGLE_APPLICATION_CREDENTIALS con il path al file json
    3) Fallback al path relativo: ../../properties/service_account.json
    """
    sa_json = os.getenv("GDRIVE_SA_JSON")
    if sa_json:
        info = json.loads(sa_json)
        return Credentials.from_service_account_info(info, scopes=SCOPES)

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa_path:
        base_dir = Path(__file__).resolve().parent
        sa_path = str((base_dir / '..' / '..' / 'properties' / 'predict-soccer-storage-cea377aa5be0.json').resolve())

    if not os.path.exists(sa_path):
        raise FileNotFoundError(
            f"Credenziali non trovate. "
            f"Imposta GDRIVE_SA_JSON o GOOGLE_APPLICATION_CREDENTIALS, "
            f"oppure crea il file: {sa_path}"
        )
    return Credentials.from_service_account_file(sa_path, scopes=SCOPES)


def _load_credentials():
    """Carica le credenziali usando il flusso OAuth2 per applicazioni desktop."""
    base_dir = Path(__file__).resolve().parent
    sa_path = str((base_dir / '..' / '..' / 'properties' / 'client_secret.json').resolve())
    SCOPES_MY_DRIVE = ['https://www.googleapis.com/auth/drive.file']
    flow = InstalledAppFlow.from_client_secrets_file(sa_path, SCOPES_MY_DRIVE)
    creds = flow.run_local_server(port=0)
    return creds


def _get_service():
    """Inizializza il servizio Google Drive in modo lazy."""
    global _service
    if _service is None:
        creds = _load_credentials()
        _service = build('drive', 'v3', credentials=creds)
    return _service


class ConfigDrive:
    """
    Utility per upload/download su Google Drive.
    - Per i modelli Python in RAM (pickle): usa upload_model / download_model
    - Per file fisici su disco: usa upload_file
    Nota: folder_id deve essere l'ID della cartella (non il nome).
    """

    @staticmethod
    def upload_file(filepath: str, folder_id: Optional[str] = None, mime_type: Optional[str] = None) -> str:
        svc = _get_service()
        metadata = {'name': os.path.basename(filepath)}
        if folder_id:
            metadata['parents'] = [folder_id]
        media = MediaFileUpload(filepath, mimetype=mime_type, resumable=True)
        file = svc.files().create(body=metadata, media_body=media, fields='id,name,parents').execute()
        return file['id']

    @staticmethod
    def upload_model(model, filename: str = 'model.pkl', folder_id: Optional[str] = None) -> str:
        """
        Serializza il modello in RAM e lo carica su Drive come blob binario.
        """
        svc = _get_service()
        data = pickle.dumps(model)
        buf = io.BytesIO(data)
        media = MediaIoBaseUpload(buf, mimetype='application/octet-stream', resumable=True)
        metadata = {'name': filename}
        if folder_id:
            metadata['parents'] = [folder_id]
        file = svc.files().create(supportsAllDrives=True, body=metadata, media_body=media,
                                  fields='id,name,parents').execute()
        return file['id']

    @staticmethod
    def download_model(file_id: str):
        """
        Scarica il blob e fa un pickle.loads() per ricostruire il modello.
        """
        svc = _get_service()
        request = svc.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        buf.seek(0)
        return pickle.loads(buf.read())


estimator = RandomForestClassifier()
config_drive = ConfigDrive()
file_id = config_drive.upload_model(estimator, folder_id='1v8DYhlHmNIwHos_zihSQZ3u4p_0FdyG-',
                                    filename='random_forest_model.pkl')
loaded_model = config_drive.download_model(file_id)
assert isinstance(loaded_model, RandomForestClassifier)
