# Test Report — Soccer Oracle

- Baseline: 17 failed, 1358 passed (`/tmp/work/soccer_baseline_final.log`).
- Dopo le modifiche: 17 failed, 1420 passed (`/tmp/work/soccer_after.log`).
- Stessi 17 fallimenti preesistenti; +62 test nuovi verdi.
- Mirati nuovi: 114 passed in ~7s (`api_sports_provider`, `scheduler`, `settlement`,
  `request_api_odds`, `serving_transparency`, `walk_forward_1x2_benchmark_test`).
