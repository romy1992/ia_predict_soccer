# Frontend React Dashboard

Frontend React (Vite) per il controllo operativo della piattaforma ML.

## Struttura FE-01 (feature-based)
- `src/App.jsx`: orchestratore stato/API
- `src/features/layout/`: sidebar, top filters, router locale
- `src/features/matches/`: tabella partite, dettaglio match, pagine match
- `src/features/data-center/`: entrypoint area operativa dati
- `src/features/ml-lab/`: entrypoint area ML lab
- `src/features/oracle/`: entrypoint area oracle picks
- `src/features/data-quality/`: dashboard qualità dati
- `src/features/shared/`: formatter e costanti comuni

## Comandi locali
```powershell
npm install
npm run dev
```

## Build produzione
```powershell
npm run build
npm run preview
```

## Variabili ambiente
Crea `frontend/.env` partendo da `frontend/.env.example`.

- `VITE_API_BASE_URL=http://localhost:8000`

## Container
Questo frontend viene containerizzato con `frontend/Dockerfile` ed esposto tramite Nginx su porta `3000` (mappata da `80` del container).



