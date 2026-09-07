export const MENU_ITEMS = [
  {
    id: "dashboard",
    label: "Dashboard",
    description:
      "Panoramica partite del giorno e live, con previsioni per mercato. Es: partite di Serie A di oggi con probabilità Over/Under 2.5 e badge PLAY/NO BET.",
  },
  {
    id: "predictions",
    label: "Storico previsioni",
    description:
      "Calcola una previsione manuale per una fixture/mercato e consulta lo storico previsioni generate. Es: fixture_id 12345, mercato '1X2'.",
  },
  {
    id: "betslip",
    label: "Schedina Oracle",
    description:
      "Genera schedine da 2/3/4 eventi (profili Safe/Balanced/Aggressive) con controllo anti-correlazione. Es: schedina Safe, quota 1.85, EV +0.12.",
  },
  {
    id: "data-center",
    label: "Data Center",
    description:
      "Importa/sincronizza dati grezzi da API-Sports: storico, oggi, calendario futuro, settlement. Es: importa Serie A dal 1 al 10 settembre.",
  },
  {
    id: "data-quality",
    label: "Data Quality",
    description:
      "Audit di sola lettura sulla qualità dei dati a DB: coverage odds, anomalie, duplicati. Es: 'BTTS' copertura odds 13% su 6392 fixture.",
  },
  {
    id: "ml-lab",
    label: "ML Lab",
    description:
      "Import giornaliero e retrain dei modelli ML sui dati già a DB. Es: retrain di tutti i mercati sulle ultime 3 stagioni.",
  },
  {
    id: "monitoring",
    label: "Monitoring",
    description:
      "Sorveglianza modelli in produzione: alert, ROI rolling, drift calibrazione, feature coverage. Es: alert ECE drift sul mercato Over 2.5.",
  },
  {
    id: "settings",
    label: "Impostazioni",
    description:
      "Attiva/disattiva job automatici e controlla la quota giornaliera API-Sports. Es: disattiva 'Sync live' per risparmiare quota.",
  },
];

