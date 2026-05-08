# MI Classification — BCI Competition IV Datasets 2a & 2b

Comparison of classical ML and deep learning methods for Motor Imagery EEG classification.

**Models:** CSP+SVM, FBCSP+SVM, EEGNet, DeepConvNet, ShallowConvNet, ATCNet, MCSANet  
**Datasets:** BCI Competition IV Dataset 2a (4-class, 22 channels) and Dataset 2b (2-class, 3 channels)  
**Protocols:** T-E holdout (subject-dependent) and LOSO (subject-independent)

---

## Requirements

```
Python 3.10+
tensorflow >= 2.10
scikit-learn
numpy
scipy
```

Install dependencies:

```bash
pip install tensorflow scikit-learn numpy scipy
```

---

## GPU Support — WSL Required on Windows

TensorFlow's GPU support on Windows requires **WSL 2 (Windows Subsystem for Linux)** with a CUDA-enabled GPU. Native Windows TensorFlow does not support GPU acceleration from TF 2.11 onwards.

### Setup (one-time)

This [YouTube tutorial](https://www.youtube.com/watch?v=LHtNv-dq8I4) is very useful for getting TensorFlow running with GPU on WSL.

If you are running on CPU only, the code will still work — deep learning models will just be slower. ATCNet and MCSANet in particular benefit significantly from GPU.

---

## Data Setup

Place the dataset files in the following structure:

```
data/
  2a/
    A01T.mat  A01E.mat
    A02T.mat  A02E.mat
    ...
    A09T.mat  A09E.mat
  2b/
    B01T.mat  B01E.mat
    B02T.mat  B02E.mat
    ...
    B09T.mat  B09E.mat
```

Dataset files can be downloaded from the [BCI Competition IV website](https://www.bbci.de/competition/iv/).

---

## Running Experiments

All experiments are run through `run_all.py`.

### Basic usage

```bash
python run_all.py --protocol te --all-subjects --dataset 2a
```

### Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | `2a`, `2b` | `2b` | Which dataset to use |
| `--protocol` | `te`, `loso`, `cv` | `te` | Evaluation protocol |
| `--all-subjects` | flag | off | Run all 9 subjects |
| `--subject` | 1–9 | `1` | Single subject (ignored if --all-subjects) |
| `--models` | see below | all | Which models to run |
| `--n-splits` | integer | `10` | Number of CV folds (cv protocol only) |

### Available models

| Name | Description |
|------|-------------|
| `csp` | CSP + SVM |
| `fbcsp` | Filter Bank CSP + SVM |
| `eegnet` | EEGNet |
| `deepconv` | DeepConvNet |
| `shallowconv` | ShallowConvNet |
| `atcnet` | ATCNet |
| `mcsanet` | MCSANet |

### Example commands

```bash
# T-E holdout, all subjects, Dataset 2a
python run_all.py --protocol te --all-subjects --dataset 2a

# LOSO, all subjects, Dataset 2b
python run_all.py --protocol loso --all-subjects --dataset 2b

# T-E, single subject, selected models
python run_all.py --protocol te --subject 3 --dataset 2a --models eegnet atcnet

# Run only classical models on Dataset 2b
python run_all.py --protocol te --all-subjects --dataset 2b --models csp fbcsp
```

### Resuming interrupted runs

Long runs (especially LOSO with all models) can take many hours. Results are saved to a checkpoint file after each subject/model completes. If a run is interrupted, re-running the same command will automatically skip already-completed results and resume from where it left off.

---

## Results

Results are saved to:
- `results_2a.txt` / `results_2b.txt` — final summary tables
- `results_2a_te_checkpoint.txt` / `results_2a_loso_checkpoint.txt` — per-subject checkpoints
- `results_2b_loso_checkpoint.txt` — Dataset 2b LOSO checkpoints

---

## Citation

If referencing this work:

```
V. Jaroenpanichying, "Comparing Classical and Deep Learning Methods for Motor Imagery
EEG Classification," BEng Robotics Engineering Final Year Project, University of Bath,
2026. (unpublished)
```

---

## Attribution

- **ATCNet** architecture adapted from [Altaheri et al. — EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) (MIT License)
- **EEGNet, DeepConvNet, ShallowConvNet** architectures referenced from [ARL EEGModels](https://github.com/vlawhern/arl-eegmodels) (Creative Commons License)
- **MCSANet** implemented from scratch following Devi et al. 2026
- **CSP / FBCSP** implemented from scratch following Ang et al. 2008
