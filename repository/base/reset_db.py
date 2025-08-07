"""
⚠️ Attenzione: cancella TUTTI i dati
In caso di necessità, droppa tutto il db cancellandolo
"""
from repository_db import engine
from service_ia.model.match import Base

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

print("Database ricreato da zero.")
