# UiAs: User-Independent 3D Facial Anti-Spoofing via Multi-modal Wireless Signals

Official anonymous implementation of UIAS, a user-independent 3D facial
anti-spoofing system based on synchronized mmWave and acoustic signals.

The repository contains signal preprocessing, the multimodal model, two-stage
training, held-out evaluation, synchronized acquisition code, a documented
data interface, and a privacy-redacted demonstration. Participant-derived
measurements, identity mappings, trained weights, and experiment outputs are
not distributed.

## Demo

The privacy-redacted demonstration is stored at `demo/demo.mp4`. Participant
and laboratory footage has been replaced with result cards, and the audio and
editor metadata have been removed.


https://github.com/Infinity1076/UiAs/blob/main/demo/demo.mp4



## Repository structure

```text
.
|-- demo/
|   `-- demo.mp4                    # privacy-redacted demonstration
|-- matlab/
|   |-- acquire_mmwave_acoustic.m   # merged acquisition and signal processing
|   `-- run.m                       # editable MATLAB entry point
|-- uias/
|   |-- dataload.py                 # data contract and group-disjoint folds
|   |-- model.py                    # multimodal UIAS architecture and losses
|   |-- preprocess_data.py          # modality-specific signal preprocessing
|   |-- train.py                    # two-stage training and checkpoint export
|   |-- test.py                     # held-out accuracy evaluation
|   `-- __init__.py
|-- pyproject.toml
|-- requirements.txt
`-- LICENSE
```

The package is intentionally flat: each module has one clear responsibility,
and the project avoids duplicate training, metric, and utility files.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

The released encoders use `weights=None`; no pretrained model is downloaded or
required.

## Data interface

Participant indices, labels, and filename mappings remain private. The local
adapter described in `uias/dataload.py` reads processed recordings and exposes
unique `group_ids`, `dataset_for_groups(group_ids)`, and
`stage1_dataset_for_groups(group_ids)`. Each dataset item returns
`input_2d [128,128]`, `input_ra [50,128]`, `acoustic [401]`, `label`,
`condition_id`, and `group_id`.



## Synchronized acquisition

Open `matlab/run.m`, edit `outputRoot`, `sampleCount`, and `startId`, then run
the file in MATLAB. The entry point calls `acquire_mmwave_acoustic.m`, which
contains both modality workers and all released signal processing.

Every sample follows the same barrier:

```text
ARM -> acoustic/mmWave ARMED -> shared GO -> acoustic/mmWave DONE
```

Outputs are written atomically using the canonical layout documented above.

Requirements are MATLAB R2024b, Parallel Computing Toolbox, Audio Toolbox,
Instrument Control Toolbox, and the TI support package for
IWR1843BOOST/DCA1000. The gate provides software-synchronized acquisition
rounds; it does not imply shared hardware-clock synchronization.



## Workflow

### 1. Preprocess

Process the synchronized recordings before loading them for training:

```bash
python -m uias.preprocess_data --input <recordings> --output <processed>
```

The input directory may be one dataset or a parent containing multiple
datasets. Each dataset contains either `mm/2d`, `mm/ra`, and `ac`, or the
legacy `-mm/2d`, `-mm/ra`, and `-ac` layout. MAT filenames are integer sample
identifiers shared by all three modalities. Existing outputs are reused unless
`--force` is supplied.

The preprocessor writes the canonical layout:

| Output | Variable | Shape |
|---|---|---:|
| `mm/2d/<sampleId>.mat` | `b` | `[128, 128]` |
| `mm/ra/<sampleId>.mat` | `doa_2d_db` | `[50, 128]` |
| `ac/<sampleId>.mat` | `energiess` | `[401, 1]` |

No participant manifest, filename mapping, or absolute input path is written
to the output.

### 2. Train

Configure the private adapter to read `<processed>`, then train Stage I and
Stage II:

```bash
python -m uias.train \
  --adapter your_private_loader:create_adapter \
  --output results
```

Training uses only the training groups in each fold. It uses fixed epoch
counts and saves `stage1_final.pt`, `model_final.pt`, and `training_log.json`
under `results/fold_<N>/`. Held-out recordings are not loaded during training
and cannot affect checkpoint selection, early stopping, or anchor updates.

### 3. Test

Load the final checkpoints and evaluate the held-out groups:

```bash
python -m uias.test \
  --adapter your_private_loader:create_adapter \
  --checkpoints results
```

Testing reports only the Accuracy of each fold and the mean Accuracy across
folds. It writes the result to `results/test_results.json`. Use the same
`--folds`, `--test-groups-per-fold`, and `--seed` values for training and
testing so that both commands reconstruct the same group-disjoint splits.



## License

Code is released under the Apache License 2.0. 
