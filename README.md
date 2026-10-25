# vibecoders — Synthetic Human Agent

An agent that uses a Windows computer the way a person does. It browses websites, reads pages, and writes and formats documents in OpenOffice Writer, using only real mouse and keyboard input. Its mouse movement and typing come from deep learning models trained on recorded human data, so the activity is hard to tell apart from a real office worker's.

Built for the NTU x ST Engineering Hackathon.

---

## Highlights

- **Learned mouse movement.** An LSTM + Mixture Density Network trained on 40,000 real browsing strokes (BMDD) generates each path: shape, speed profile and timing. Click hold and dwell times are learned too.
- **Learned typing.** A second LSTM-MDN trained on 3.8M keystrokes from 5,367 typists (Aalto) sets key timing, hold times and rollover. Typos and their corrections come from the same data. Thinking pauses, false starts and revisions are learned from 2,471 essays (KLiCKe).
- **A different person every run.** A sampled persona sets typing speed, mouse speed, shortcut-vs-mouse preference, reading speed and how long the person thinks. Fatigue builds and tempo drifts over the session.
- **Real input only.** No file generation, COM/UNO automation or browser injection. Pages are read through Windows UI Automation, the accessibility interface screen readers use.
- **Offline and lightweight.** No LLM and no internet needed. The models are trained in TensorFlow but run in pure numpy, using about 100 MB of RAM on a 4 GB machine.

| Result | Value |
|---|---|
| Typing: real-vs-generated detector AUC (0.5 = can't tell) | **0.52** |
| Mouse: KS distance to real strokes (peak speed / deviation / duration) | **0.06 / 0.08 / 0.11** |
| Mouse: strokes reaching the target | 599 / 600 |
| Research run: typed blocks that matched the plan exactly | 20 / 20 |

---

## Requirements

- Windows 10 or 11, Python 3.10+
- Google Chrome
- Apache OpenOffice 4.1 (Writer); built and tested with 4.1.16 en-GB
- English Windows display language recommended. Button and menu names are matched in English; other languages fall back to keyboard shortcuts.

```bash
pip install -r requirements.txt
```

The trained models in `models/` (`mouse_mdn.npz`, `typing_mdn.npz` and the `*.json` statistics) are all the agent needs at runtime. Without them it falls back to simpler built-in generators.

Check that a machine is ready before a run:

```bash
python -m tools.preflight            # static checks; nothing is opened or changed
python -m tools.preflight --launch   # also opens Writer and Chrome briefly to check UI Automation
```

---

## Usage

All modes run through `main.py`. **To abort at any time, slam the mouse into a screen corner.**

### Normal mode: the 30-minute office worker
15 minutes browsing the allowed websites, then 15 minutes of office work:

- **Notepad as scratch space.** While browsing, the agent sometimes copies a page's address from Chrome and pastes it into a note with a word on why. Later it jots the budget totals there too.
- **OpenOffice Calc budget sheet** (`--sheet-minutes`, default 4, taken from the writing time). Last month's expenses are typed in and totalled with `SUM`. A `SUMPRODUCT` summary per category is typed once and copied down to the other categories. Headers and totals are bolded, amounts get two decimals, and a column chart of the summary is added with Insert > Chart. Often the columns are set to their optimal width, the top row frozen and the sheet renamed. Saved as `.ods`. OpenOffice shows UI Automation nothing, so the agent reads Calc's grid, column headers and menu bar from a screenshot (`controller/calc_screen.py`, numpy only). Mouse-minded personas then click cells, drag across ranges, open menus by their titles and fit columns by double-clicking a header border. Each click is checked on a fresh screenshot, with the keyboard as the fallback. While typing, the mouse drifts to the row being worked on instead of sitting still.
- **Writer report** (status report, meeting notes, proposal or memo). Its spending section quotes the sheet's totals. The summary table is copied from Calc, pasted as unformatted text and turned into a Writer table with Table > Convert > Text to Table. If that can't be done, the table is typed. Writer is read from screenshots too (`controller/writer_screen.py`: the page, its lines of text, the menu bar). Mouse-minded personas open menus by their titles (Insert > Table, Table > Convert) and highlight the line just typed by dragging or triple-clicking, each checked on a fresh screenshot. The mouse also drifts to the text while typing and over dialogs while reading them.

The report, the spreadsheet and `notes.txt` are saved under `runs/normal/`.

```bash
python main.py --process normal
python main.py --process normal --target-sites 10 --target-sentences 30
python main.py --process normal --browse-minutes 2 --write-minutes 2 --target-sites 2 --target-sentences 5   # quick test
```

### Research mode
Searches a topic and takes notes from every page it reads. It then writes a cited research paper (title, abstract, headings, tables, bullets, references) into Writer. Output goes to `runs/research/<topic>-<time>/`.

```bash
python main.py --process research --topic "Apollo program"
python main.py --process research --topic "Apollo program" --write-minutes 30
python main.py --process research --resume runs/research/apollo-program-20261001-1030
```

### Other modes

```bash
python main.py                                    # Chrome search demo
python main.py --process writer --wpm 80          # type a paragraph into Writer as an 80 wpm typist
python main.py --process session --minutes 15     # mixed browsing, Writer, Excel and Notepad with idle breaks
python main.py --process compose --notes tests/fixtures/pages --topic "Apollo program" --dry-run   # offline, no UI
```

### Useful options

| Flag | Meaning |
|---|---|
| `--seed N` | Reproduces the whole run, persona included |
| `--wpm`, `--pause-scale`, `--temperature`, `--shortcut-pref` | Override single persona traits |
| `--no-fatigue` | Turn off fatigue and tempo drift |
| `--no-model` | Use the fallback mouse generator instead of the trained model |
| `--sites-file FILE` | Use your own list of allowed websites |
| `--profile NAME` | Chrome profile to pick in the "Who's using Chrome?" picker |

Run `python main.py --help` for the full list.

---

## Project structure

```
main.py              entry point and demos
modes.py             normal and research modes (browsing phase + writing phase)
tasks.py             task building blocks (open apps, search, close Writer, ...)
surfing.py           browsing: site visits, link following, site search
sites.py             allowed websites and how to search them
office_writing.py    generated office documents for normal mode

algorithms/          HOW a person does things
  behaviour.py         Persona and Human: thinking time, method variety, fatigue
  human_mouse.py       learned mouse movement and clicks
  human_typing.py      learned typing: rhythm, typos, pauses, revisions
  reading.py           reading and scrolling
  mouse_model/         LSTM-MDN mouse model (training + numpy runtime)
  typing_model/        LSTM-MDN typing model and statistical models

controller/          WHAT can be done: low-level input and app control
  mouse.py, keyboard.py, win_input.py    input
  screen.py, ui_elements.py              finding things on screen (image matching, UI Automation)
  apps.py, browser.py, office.py, writer.py

research/            research pipeline: page extraction, notes, offline paper composition,
                     Writer operations and self-checks

models/              trained models used at runtime
tools/               preflight check, Writer smoke test, page capture, probes
tests/               offline unit tests and fixtures
```

---

## Training the models (optional)

The trained models are already in `models/`. To retrain, put the datasets in `data/`. They are multi-GB and git-ignored.

```bash
# mouse model (BMDD; the split archive is read directly, no extraction needed)
python -m algorithms.mouse_model.train --data data/BMDD/boun-mouse-dynamics-dataset.zip --out models --window-filter browsing --max-strokes 40000 --epochs 25
python -m algorithms.mouse_model.evaluate --data data/BMDD/boun-mouse-dynamics-dataset.zip --window-filter browsing

# typing model (Aalto keystrokes + KLiCKe essays)
python -m algorithms.typing_model.fit_stats --aalto data/aalto_keystrokes/Keystrokes.zip --klicke data/klicke/linking-writing-processes-to-writing-quality.zip
python -m algorithms.typing_model.train --data data/aalto_keystrokes/Keystrokes.zip
```

Training needs the extra packages listed (commented out) in `requirements.txt`: TensorFlow, pandas, scipy, scikit-learn and matplotlib.

---

## Testing

```bash
python -m unittest discover -s tests -t .            # offline tests
python -m tools.writer_smoke --shortcut-pref 0.05    # ~3 min live check of every Writer formatting type
```

---

## Documentation

| Document | Contents |
|---|---|
| [ALGORITHMS.md](ALGORITHMS.md) | Behaviour layer: mouse and typing models, personas, reading, evaluation results |
| [RESEARCH.md](RESEARCH.md) | Research mode: page reading, offline writing, Writer automation, live-testing findings |
| [CONTROLLER.md](CONTROLLER.md) | Low-level controller function reference |
| [algorithms/mouse_model/README.md](algorithms/mouse_model/README.md) | Mouse model in depth |
| [algorithms/typing_model/README.md](algorithms/typing_model/README.md) | Typing model in depth |

---

## Datasets

- **BMDD**: Boğaziçi University Mouse Dynamics Dataset (mouse movement)
- **Balabit Mouse Dynamics Challenge** (mouse and scroll events)
- **Aalto 136M Keystrokes** (typing timing and typos)
- **KLiCKe** / *Linking Writing Processes to Writing Quality* (pauses and revisions while composing)
