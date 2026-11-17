# Python in my Backend: ball possession from tracking data

Our entry for **More than a Hack 2027, Phase 1 (Ball Possession)**. The pipeline labels every frame of the
FC Barcelona v Athletic Club tactical video as `H` (Barcelona), `A` (Athletic Club) or `D` (dead
ball), using only the automatic tracking provided with the challenge.

- Submission file: [`submission/Python in my Backend.csv`](submission/)
- Technical document: [`docs/Python in my Backend.pdf`](docs/)

## How it works

1. **Football rules** (`rules_v2.py`). These detect touches, control, dead balls, restarts and
   balls out of play from player and ball positions.
2. **Two sequence models** (`models/`). A bidirectional GRU and a dilated temporal CNN read 52
   per-frame features, about 20 s either side of each frame. They were trained on 7 Bundesliga
   matches from the open DFL/IDSSE dataset. The clean tracking there was degraded so that it looks
   like ours: missing ball, missing players, mostly goalkeepers.
3. **Decoding** (`decode.py`). The two models' probabilities are averaged, then smoothed with a Viterbi
   pass that charges a cost for each change of possession. A frame is dead if either the rules or
   the models say so. Possessions shorter than 1 s are removed.

The technical document covers the details, the possession definition we used and the limitations.

| | spellF1 | macroF1 |
|---|---|---|
| Public 5-minute window | 0.625 | 0.811 |
| DFL matches, leave-one-match-out (models only) | 0.599 | – |

## Reproducing the submission

Requirements: Python 3.13 and the challenge data bundle. No GPU is needed for inference. The
first run builds the feature cache and takes about a minute on a laptop.

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027
cd more-than-a-hack-2027
# put the challenge data bundle in ./data as described in the challenge DATA.md
git clone https://github.com/chatrapalsr/more-than-a-hack-2027-ball-possession solution
pip install -r solution/requirements.txt
python solution/make_submission.py
python evaluation/validate_submission.py "solution/submission/Python in my Backend.csv"
```

The code uses the challenge repository's loaders (`games.py`, `evaluation/`), which is why it
lives in `./solution` inside it. We checked this from a fresh clone with no caches (Windows 11,
Python 3.13.7, CPU only): the output is byte-for-byte identical to the committed CSV.

To produce the variant chosen only on the DFL matches (models also decide dead balls), run
`python solution/make_submission.py --mode model`.

## Retraining (optional)

The trained weights are in `models/`. To retrain them:

```bash
python solution/bar_profile.py        # tracking-damage profile of the provided match (no labels)
python solution/opendata.py convert   # downloads and converts the 7 DFL matches (~2.5 GB)
```

Then run `training/run_d.py` (GRU) and `training/run_e.py` (TCN) on a GPU machine; each takes
about 25 minutes on a single T4. `training/kprep.py` downloads the data and builds the features
for those scripts. `training/harness_ensemble.py` scores the saved out-of-fold predictions, and
`training/tune.py` is the parameter sweep we used for the rules.

## Layout

```
make_submission.py      end-to-end: data -> submission CSV
rules_v2.py             football rules (touches, control, dead ball, restarts, out of play)
rules_pipeline.py       data loading and first rule version
features.py             52 per-frame features
decode.py               Viterbi decoding and minimum run lengths
predict_learned.py      model inference
opendata.py             DFL/IDSSE conversion, tracking damage, label relabelling
bar_profile.py          tracking-damage statistics of the provided match
models/                 trained GRU (d) and TCN (e) weights
training/               training, tuning and evaluation scripts
submission/             the submitted CSV
docs/                   technical document
```

## Data and licences

- **Training data:** Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of
  spatiotemporal and event data in elite soccer.* Scientific Data. DFL/IDSSE open data, CC BY 4.0.
- **Challenge match data:** not included here. It is licensed for the competition only.
- **Code:** MIT licence, see [LICENSE](LICENSE).
