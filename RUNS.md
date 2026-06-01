# Reproduction Commands

All commands are meant to be run from the repository root after installing
`requirements.txt`. The archived CSV and report files under `outputs/` were
generated with local CPU runs.

## Raw-Index Confirmatory Runs

```powershell
.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models mlp_raw_2x512 --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.001 --weight-decay 0.0001 --activation silu --output-dir outputs\confirm_mlp2x512_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models mul_mlp_raw_2x256 --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.00001 --weight-decay 0.0 --activation silu --output-dir outputs\confirm_product_mlp2x256_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models multkan_raw_medium --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.001 --weight-decay 0.0 --activation silu --output-dir outputs\confirm_multkan_medium_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models fourier_mlp_raw,siren_raw --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.001 --weight-decay 0.0001 --hidden-dim 128 --depth 3 --activation silu --output-dir outputs\confirm_feature_controls_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models nac_raw,nalu_raw --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --hidden-dim 64 --depth 2 --lr 0.001 --weight-decay 0.0 --grad-clip 10.0 --output-dir outputs\confirm_nac_nalu_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models iter_rnn_raw_8 --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.001 --weight-decay 0.0 --grad-clip 10.0 --output-dir outputs\confirm_iter_rnn_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models iter_gru_raw_8 --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.001 --weight-decay 0.0 --grad-clip 10.0 --output-dir outputs\confirm_iter_gru_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models iter_lstm_raw_8 --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.001 --weight-decay 0.0 --grad-clip 10.0 --output-dir outputs\confirm_iter_lstm_10seed_1m_eval10k

.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models linear_recurrence_raw_2 --seeds 0,1,2,3,4,5,6,7,8,9 --steps 1000000 --eval-every 10000 --lr 0.0001 --weight-decay 0.0 --output-dir outputs\confirm_recurrent_h2_10seed_1m_eval10k
```

## Analytic Diagnostics

```powershell
.\.venv\Scripts\python.exe -m fibonacci_grokking.index_study --models log_linear,poly_raw --seeds 0 --steps 0 --eval-every 1 --output-dir outputs\short_raw_analytic_baselines
```

## Dynamics Plot

```powershell
.\.venv\Scripts\python.exe scripts\plot_short_dynamics.py --curves outputs\confirm_mlp2x512_10seed_1m_eval10k\curves.csv outputs\confirm_product_mlp2x256_10seed_1m_eval10k\curves.csv outputs\confirm_multkan_medium_10seed_1m_eval10k\curves.csv outputs\confirm_feature_controls_10seed_1m_eval10k\curves.csv outputs\confirm_nac_nalu_10seed_1m_eval10k\curves.csv outputs\confirm_iter_rnn_10seed_1m_eval10k\curves.csv outputs\confirm_iter_gru_10seed_1m_eval10k\curves.csv outputs\confirm_iter_lstm_10seed_1m_eval10k\curves.csv outputs\confirm_recurrent_h2_10seed_1m_eval10k\curves.csv --output outputs\short_dynamics.pdf
```

## Modular Controls

For each encoding in `onehot`, `scalar`, and `fourier`, and each seed in
`0,1,2,3,4`, run:

```powershell
.\.venv\Scripts\python.exe -m fibonacci_grokking.train --model mlp --task all_pairs --modulus 17 --encoding scalar --seed 0 --steps 50000 --eval-every 1000 --batch-size 0 --lr 0.001 --weight-decay 1.0 --hidden-dim 256 --depth 2 --activation gelu --output outputs\modular_m17_scalar_seed0_50k.csv

.\.venv\Scripts\python.exe -m fibonacci_grokking.train --model mlp --task all_pairs --modulus 7 --encoding scalar --seed 0 --steps 30000 --eval-every 1000 --batch-size 0 --lr 0.001 --weight-decay 1.0 --hidden-dim 256 --depth 2 --activation gelu --output outputs\modular_m7_scalar_seed0_30k.csv
```

Replace `scalar` and `seed0` in the example paths with the selected encoding
and seed.
