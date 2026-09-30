# Datasets

## Bogazici Mouse Dynamics Dataset (BMDD) — primary (in use)

- **What:** 24 users, about 2,550 hours of normal computer use, recorded natively on Windows by an OS-level listener. It is 33.8 GB uncompressed, in 157,456 session files.
- **License:** CC BY 4.0. Cite: Kılıç, Yıldırım et al., *"Bogazici mouse dynamics dataset"*, Data in Brief (2021).
- **Source:** https://data.mendeley.com/datasets/w6cxr8yc7p/2
- **Where it is:** the split archive `data/BMDD/boun-mouse-dynamics-dataset.zip` plus `.z01`–`.z09`. **No extraction needed.** Pass the `.zip` path as `--data`, and files are streamed from the archive by `algorithms/mouse_model/splitzip.py`.

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
python -m algorithms.mouse_model.splitzip list data/BMDD/boun-mouse-dynamics-dataset.zip
python -c "from algorithms.mouse_model.dataset import peek; peek('bmdd', 'data/BMDD/boun-mouse-dynamics-dataset.zip')"
# optional: extract just the training files (~3.3 GB) if you prefer a plain folder
python -m algorithms.mouse_model.splitzip extract data/BMDD/boun-mouse-dynamics-dataset.zip --pattern /training/ --dest data/bmdd
```

## Balabit Mouse Dynamics Challenge — secondary (optional)

- **What:** 10 users, captured from remote-desktop (RDP) traffic. It is smaller and lower quality.
- **Download:**
  ```bash
  git clone https://github.com/balabit/Mouse-Dynamics-Challenge data/balabit
  ```
- Columns: record timestamp, client timestamp, button, state, x, y. Session files have no extension.

## Aalto 136M Keystrokes — typing rhythm and typos (in use)

- **What:** 168,000 volunteers each copied 15 sentences in an online typing test (136M keystrokes). Each keystroke has its press and release time (ms), the key and the keycode. The target sentence and the final typed text are stored with it, so every typo and correction can be labelled.
- **License:** free for **non-commercial** use with attribution. Cite: Dhakal, Feit, Kristensson, Oulasvirta, *"Observations on Typing from 136 Million Keystrokes"*, CHI 2018.
- **Source:** https://userinterfaces.aalto.fi/136Mkeystrokes/ (`Keystrokes.zip`, 1.5 GB; 16 GB unzipped).
- **Where it is:** `data/aalto_keystrokes/Keystrokes.zip`. **No extraction needed**: `algorithms/typing_model/dataset_aalto.py` reads participant files straight from the zip.

Layout inside the archive:
```
Keystrokes/files/<PARTICIPANT_ID>_keystrokes.txt   (tab separated, one file per typist)
Keystrokes/files/metadata_participants.txt         (layout, keyboard type, WPM, error rate, fingers, ...)
Keystrokes/files/readme.txt
```
Columns: `PARTICIPANT_ID, TEST_SECTION_ID, SENTENCE, USER_INPUT, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE`. `LETTER` is the character, or a key name (`BKSP` / `\x08`, `SHIFT`, `CAPS_LOCK`, `ARW_LEFT`, ...).

Quirks handled by the loader:
- About 9% of typists' browsers reported an empty `LETTER`; those trials are skipped.
- Trials that use arrows, Delete or Ctrl are skipped, because they can't be replayed as append/Backspace.
- Only QWERTY desktop/laptop keyboards are used.
- About 30% of browsers recorded timestamps on a 4/8/16 ms grid; `evaluate.py` matches this when comparing.

## KLiCKe keystroke logs (Kaggle "Linking Writing Processes to Writing Quality") — pauses and revisions (in use)

- **What:** about 2,500 argumentative essays written freely in 30 minutes, 8.4M events. Each event has key-down/up times, the activity (Input, Remove/Cut, Nonproduction, Replace, Paste, Move), the text change, the cursor position and the word count. **Letters and digits are masked as `q`**; spaces, punctuation and newlines are kept.
- **License / access:** Kaggle competition data. Log in, accept the competition rules, and download from https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data
- **Where it is:** `data/klicke/linking-writing-processes-to-writing-quality.zip`, read in place (or an extracted `train_logs.csv`).
- Only `train_logs.csv` (486 MB) is used. `test_logs.csv`, `train_scores.csv` and `sample_submission.csv` in the same zip are not needed.

Fit / train commands (writes to `models/`):
```bash
python -m algorithms.typing_model.fit_stats --aalto data/aalto_keystrokes/Keystrokes.zip --klicke data/klicke/linking-writing-processes-to-writing-quality.zip
python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip
```

## Optional typing datasets (not used yet)
- **CoAuthor** (https://coauthor.stanford.edu): 1,445 writing sessions with real, unmasked text insert/delete events. Useful to learn *what* people rewrite. Filter out the GPT-3 suggestion events.
- **Buffalo free-text** and **Clarkson II** keystroke datasets (on request from the authors): extra validation that the timing generalises beyond copying.
