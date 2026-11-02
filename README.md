# Machine Learning Systems Labs

Lab work for the Machine Learning Systems course, following the course lab manual.
Textbook: V. J. Reddi, *Introduction to Machine Learning Systems*.

**Author:** Fatih Alkan

## Setup

Python 3.11.8 is used for all labs.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Repository layout

```
README.md
requirements.txt
.gitignore
data/            # not tracked, created by the scripts when needed
lab01/
    report.md
    src/
    results/
...
```

## Labs

| Lab | Topic | Folder |
|-----|-------|--------|
| 01 | Environment and first system measurements | [lab01](lab01/) |

## Reproducibility

- Fixed seeds: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- Timings: one warm-up run, then at least 5 repetitions; the median is reported. Single-sample inference is repeated 100 times.
- Every lab saves the Python and library versions to `results/versions.txt`.
