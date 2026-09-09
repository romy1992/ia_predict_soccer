import sys
print("STARTING", flush=True)
from src.service_ia.training.train_multi_market import train_market
print("IMPORT OK", flush=True)
r = train_market(market='under_over_1_5', selection_method='kbest', save_model=False)
print('STATUS:', r.status, flush=True)
print('CHAMPION:', r.champion, flush=True)
print('ROWS:', r.rows, flush=True)
print('BEST_CV_F1:', r.best_cv_f1, flush=True)
print('DETAILS KEYS:', list(r.details.keys()) if r.details else None, flush=True)
