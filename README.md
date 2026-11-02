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

### Known dependency conflict

The pinned versions in the manual are not fully consistent: `mlflow==2.14.1` declares
`pyarrow<16`, while the manual pins `pyarrow==16.1.0`. A plain `pip install -r requirements.txt`
stops with `ResolutionImpossible`. The exact pinned versions can still be installed in three steps
(Windows PowerShell shown; the same idea works on Linux/macOS with `grep -v`):

```powershell
(Get-Content requirements.txt) -notmatch '^mlflow' | Set-Content req-tmp.txt
.venv\Scripts\python.exe -m pip install -r req-tmp.txt
.venv\Scripts\python.exe -m pip install mlflow==2.14.1
.venv\Scripts\python.exe -m pip install --no-deps pyarrow==16.1.0
Remove-Item req-tmp.txt
```

After this, every package matches `requirements.txt` (check with `lab01/src/print_versions.py`).
`pip check` will still report the mlflow/pyarrow mismatch; it does not affect the labs.

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
| 02 | A neural network from scratch (NumPy) | [lab02](lab02/) |

## Reproducibility

- Fixed seeds: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
- Timings: one warm-up run, then at least 5 repetitions; the median is reported. Single-sample inference is repeated 100 times.
- Every lab saves the Python and library versions to `results/versions.txt`.
