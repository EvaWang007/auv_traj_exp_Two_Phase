NOAA Two-phase CPI-LSTM: phase-one offline pretraining

2026-11-24 formal GPU release: read SERVER_PHASE1_30EP.txt first.
New entry: bash server/run_phase1.sh (Ubuntu; current image 2.8.0+cu128/Python3.12,
legacy 2.7.1+cu118/Python3.10 also accepted). Current-image instructions:
SERVER_RTX3090_IMAGE.txt and bash server/setup_image.sh.
LSTM, CPI-LSTM, PSO-CPI-LSTM each train 30 final epochs; resumable PSO search
and frozen evaluation are included. Historical scripts below remain CPU-only.
SERVER_TRAINING.txt describes the older CPU migration path, not the new launcher.

Default environment: E:/Anaconda_envs/envs/auv_traj_exp/python.exe

Prepare data (creates data/, refuses to overwrite it):
  python prepare_phase1.py
Train 20 epochs (creates results/phase1_seed42/, refuses to overwrite it):
  python train_phase1.py

1000 normal-position-feedback trajectories driven by existing 60x NOAA currents.
Train/validation/test: 700/100/200 trajectories in disjoint NOAA source time blocks.
History: 24 s. Candidate horizon: 16 s. Window stride: 10 samples (2 s).
History LSTM: 128. Candidate-control LSTM: 96. Seed: 42.
Loss: position + 0.1 current + 0.2 position bias + 0.02 physics. No smoothness.

The complete normal records contain simulated true positions, DR, controls,
NOAA currents, true vehicle states and noisy reliable position observations.
Training windows mix normal observations and simulated outages. Requested
outage durations are capped by available history since the first fix; both
actual age and requested mode are recorded in data/windows.csv.

Only available fixes are inputs. The last-fix current prior is derived from
past fix-minus-DR differences, not from hidden NOAA current labels. When too
little fix history exists, a training-only mean current is used.

Future inputs are precommitted acceleration/yaw-rate plans generated from
the last historical command plus independently seeded exploration. Future
actual normal-controller commands are not inputs. Each plan has a new
physics rollout label, starting from the true state at the window anchor.

Current and present-position-bias heads depend on historical inputs only.
Candidate controls affect a bounded learned water-motion correction and
the lagged physics integration. They cannot change the current position.

Phase 1 is offline masked-data pretraining. Histories still come from a
normal-position-feedback controller; hiding its fixes does not recreate
the actions of a controller operating without fixes. Phase 2 must collect
model-controlled outage rollouts and evaluate actual waypoint tracking.

Results include best/final checkpoints, training curves, scalers, full
test predictions, metrics by actual outage duration, and information-boundary
checks. No actual outage-controller performance is claimed in phase 1.
