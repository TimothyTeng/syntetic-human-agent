# Datasets

## Bogazici Mouse Dynamics Dataset (BMDD) — primary (in use)

- **What:** 24 users, about 2,550 hours of normal computer use, recorded natively on Windows by an OS-level listener. It is 33.8 GB uncompressed, in 157,456 session files.
- **License:** CC BY 4.0. Cite: Kılıç, Yıldırım et al., *"Bogazici mouse dynamics dataset"*, Data in Brief (2021).
- **Source:** https://data.mendeley.com/datasets/w6cxr8yc7p/2
- **Where it is:** the split archive `boun-mouse-dynamics-dataset.zip` plus `.z01`–`.z09`, in the project root. **No extraction needed.** Pass the `.zip` path as `--data`, and files are streamed from the archive by `algorithms/mouse_model/splitzip.py`.

Layout inside the archive:
```
boun-mouse-dynamics-dataset/
  labels.csv
  users/userN/training/session_*.csv          <- used for training (~190 files, 3.3 GB)
  users/userN/internal_tests/session_*.csv    <- used by evaluate.py (--split test)
  users/userN/external_tests/session_*.csv
```

Columns: `client_timestamp` (epoch s), `x`, `y`, `button` (empty / Left / Right), `state` (Move / Drag / Pressed / Released), `window` (an anonymised category: `browsing`, `development`, `file system`, ...). Samples are about 11 ms apart.

Useful commands:
```bash
python -m algorithms.mouse_model.splitzip list boun-mouse-dynamics-dataset.zip
python -c "from algorithms.mouse_model.dataset import peek; peek('bmdd', 'boun-mouse-dynamics-dataset.zip')"
# optional: extract just the training files (~3.3 GB) if you prefer a plain folder
python -m algorithms.mouse_model.splitzip extract boun-mouse-dynamics-dataset.zip --pattern /training/ --dest data/bmdd
```

## Balabit Mouse Dynamics Challenge — secondary (optional)

- **What:** 10 users, captured from remote-desktop (RDP) traffic. It is smaller and lower quality.
- **Download:**
  ```bash
  git clone https://github.com/balabit/Mouse-Dynamics-Challenge data/balabit
  ```
- Columns: record timestamp, client timestamp, button, state, x, y. Session files have no extension.
