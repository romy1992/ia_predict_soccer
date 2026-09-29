"""Ricompone i file `.pkl` dei 5 champion (4 Under/Over + goal_no_goal) a
partire dai pezzi trasferiti via git in `scripts/analysis/_export/model_chunks/`
(i modelli superano il limite di dimensione per un allegato in chat - fino
a 101MB il singolo file - quindi sono stati spezzati in pezzi da ~45MB).

DA ESEGUIRE IN LOCALE dopo `git pull`: nessun addestramento, nessun accesso
DB - solo lettura e concatenazione di file gia' presenti nel repo, pochi
secondi. Scrive i file ricomposti in `best_models/` (sia come
`{market}_champion.pkl` sia come `{market}_champion_calibrator.pkl` - i due
file sono identici byte per byte nella pipeline di produzione attuale,
quindi la stessa ricomposizione viene copiata su entrambi i nomi).
"""
from __future__ import annotations

import os
import re
import shutil

CHUNKS_DIR = os.path.join("scripts", "analysis", "_export", "model_chunks")
OUTPUT_DIR = "best_models"
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5", "goal_no_goal"]


def main() -> None:
    if not os.path.isdir(CHUNKS_DIR):
        raise SystemExit(f"Cartella non trovata: {CHUNKS_DIR} - hai fatto 'git pull' prima di eseguire questo script?")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pattern = re.compile(r"^(.+_champion\.pkl)\.part(\d+)$")

    chunks_by_market: dict[str, list[str]] = {}
    for filename in os.listdir(CHUNKS_DIR):
        match = pattern.match(filename)
        if not match:
            continue
        base_name = match.group(1)
        chunks_by_market.setdefault(base_name, []).append(filename)

    for market in MARKETS:
        base_name = f"{market}_champion.pkl"
        chunks = sorted(chunks_by_market.get(base_name, []))
        if not chunks:
            print(f"ATTENZIONE: nessun chunk trovato per {market}, salto")
            continue

        output_path = os.path.join(OUTPUT_DIR, base_name)
        with open(output_path, "wb") as out:
            for chunk_name in chunks:
                with open(os.path.join(CHUNKS_DIR, chunk_name), "rb") as chunk_file:
                    out.write(chunk_file.read())

        size = os.path.getsize(output_path)
        print(f"{market}: {len(chunks)} chunk -> {output_path} ({size} bytes)")

        calibrator_path = os.path.join(OUTPUT_DIR, f"{market}_champion_calibrator.pkl")
        shutil.copyfile(output_path, calibrator_path)
        print(f"  copiato anche in {calibrator_path}")

    print("\nFatto. Puoi rimuovere scripts/analysis/_export/model_chunks/ se vuoi liberare spazio.")


if __name__ == "__main__":
    main()
