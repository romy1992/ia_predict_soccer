# Frontend React Dashboard

Frontend React (Vite) per il controllo operativo della piattaforma ML.

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

