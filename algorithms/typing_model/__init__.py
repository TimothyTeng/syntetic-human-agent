"""
Learned human-typing model.

Runtime (numpy only):
    planner.plan_keystrokes   text -> keystroke plan (typos, corrections, false starts,
                              revisions, pauses) with press/hold timings
    timing.BigramTimer        empirical per-bigram key timing (fallback motor model)
    sampler.TypingMDN         LSTM-MDN motor model (numpy forward pass)
    errors.ErrorStats         typo rates / types / detection + correction behaviour
    composition.CompositionStats  pauses, bursts, revision rates from essay writing
    alternatives              where "change of idea" wording comes from (heuristic / LLM)
    simulate.apply_plan       replay a plan into a virtual text buffer (for tests)

Fitting / training (offline):
    fit_stats.py   -> models/typing_bigrams.json, typing_errors.json, typing_composition.json
    train.py       -> models/typing_mdn.npz          (TensorFlow, training only)
    evaluate.py    real vs synthetic comparison

Data: dataset_aalto.py (Aalto 136M keystrokes), dataset_klicke.py (Kaggle KLiCKe logs)
"""
