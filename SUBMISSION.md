## Inspiration
Most "bots" are easy to spot. The mouse jumps in straight lines, keys arrive at perfectly even intervals, and nobody ever makes a typo or stops to think. We wanted to build the opposite: an agent whose activity on a real Windows desktop can't be told apart from a real office worker's. That's useful anywhere you need believable human activity, such as cyber ranges, training environments and testing detection systems. We also set ourselves a hard rule: learn every behaviour we could from recorded human data instead of hand-tuning it.

## What it does
vibecoders is a **Synthetic Human Agent**. It sits at a real Windows PC and works like a person for 30 minutes, using only real mouse and keyboard input.

- **Normal mode:** it spends 15 minutes browsing a set of allowed websites. It reads pages at its own pace, re-reads now and then, follows links and uses each site's search. It then closes Chrome and spends 15 minutes writing an office document (status report, meeting notes, proposal or memo) in OpenOffice Writer, with headings and formatting. It saves the document and closes Writer.
- **Research mode:** it searches a topic and takes notes from every article it reads. It then writes a cited research paper from those notes, with a title, abstract, headings, tables, bullets and references, and types it into Writer.
- **Every run is a different person.** A sampled persona sets typing speed, mouse speed, shortcut-vs-mouse preference, reading speed and how long the person thinks. On top of that, fatigue builds and tempo drifts over the session, so the person gets a little slower and sloppier as time goes on.
- **It checks its own work.** It reads the document back through UI Automation and confirms the typed text matches what it meant to type.

## How we built it
- **Mouse model:** an LSTM with a Mixture Density Network head, trained on 40,000 real browsing strokes from the BMDD dataset. It generates the whole path: its shape, speed profile and timing, overshoots included. Click hold time and dwell before a click are also learned from the data.
- **Typing model:** a second LSTM-MDN trained on 3.8M keystrokes from 5,367 typists in the Aalto dataset. It handles key timing, hold times, rollover and Shift timing. Typos, when they get noticed and how they get corrected come from the same data. Thinking pauses, false starts and in-text revisions are learned from 2,471 essays in the KLiCKe writing-process dataset.
- **Human-factors layer:** pauses come from the Keystroke-Level Model (mental preparation, the hand moving between mouse and keyboard). Choosing from a list uses Hick–Hyman visual-search time. The agent also varies its methods, e.g. sometimes the Back button and sometimes Alt+Left.
- **Offline writing:** the target machine has 4 GB of RAM and almost no internet, so there's no LLM. Papers are written extractively with TF-IDF, TextRank, MMR and k-means in plain numpy, and every sentence is cited.
- **Lean runtime:** we train in TensorFlow but run in pure numpy. That brings RAM from about 500 MB down to about 100 MB, and one mouse path takes about 10 ms to generate.
- **Real input only:** no file generation, COM/UNO automation or browser injection. Pages are read through UI Automation, the same accessibility interface screen readers use.

## Challenges we ran into
- **Autocorrect fighting the typing model.** We traced occasional missing letters to the word processor's AutoCorrect fixing a typo before our model's own correction landed. We fixed it by planning typos that never cross a word boundary.
- **Switching word processors mid-project.** Moving from Microsoft Word to OpenOffice Writer meant re-learning everything by probing live. Toggle buttons never report their state, the focus API always returns the window, Ctrl+B leaves a hidden "not bold" that carries over Enter, and shortcuts differ. We found 13 behaviours like these and handled each one.
- **Accessibility trees going stale** while pages and documents redraw, plus Chrome building its tree only after the page loads.
- **Time budgeting.** Typing at a realistic 20 wpm while finishing and saving inside a fixed 15-minute window took per-block time estimates and reserved time for saving and closing.

## Accomplishments that we're proud of
- **Typing a detector can't distinguish from real people:** a gradient-boosting classifier scored **AUC 0.52** on our generated typing vs. real typists (0.5 = chance). Our first placeholder scored 1.00.
- **Mouse movement close to real:** KS distances to held-out real strokes are 0.06 for peak speed, 0.08 for path deviation and 0.11 for duration. The model reaches its target on 599 of 600 strokes.
- **Accurate typing despite typos:** 20 of 20 blocks matched in the final full research run, with zero formatting problems, and the model still makes and corrects typos along the way.
- **Small footprint:** everything runs offline on a 4 GB machine at about 100 MB of RAM.

## What we learned
- Imitation learning on recorded human data works far better than hand-made curves or RL. There's no natural reward for "moves like a human", but there is plenty of human data.
- Predicting a full distribution (an MDN) and sampling from it matters. A model that only predicts the average is too regular and easy to detect.
- Real apps are messy. Most of our bugs came from undocumented application behaviour, not from our models, and live testing found problems unit tests never would.

## What's next for vibecoders
- Learn scroll and reading behaviour from data (the Balabit dataset has scroll events) to replace the current hand-set values.
- Fix the mouse model's sample spacing, which is about 40% too sparse, with more training data and a larger network.
- Add more tasks, such as email and spreadsheets, behind a scheduler that chains them into a full workday.
- Add a small offline paraphraser so research writing goes beyond selecting and lightly rephrasing source sentences.
