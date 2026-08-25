"""Preprocess recorded mmWave and acoustic MAT files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


MMWAVE_PEAK_RATIO = 0.95
CIR_PEAK_HALF_WINDOW = 20
MAT_KEYS = {"2d": ("b",), "ra": ("doa_2d_db",), "ac": ("energiess",)}
OUTPUT_KEYS = {"2d": "b", "ra": "doa_2d_db", "ac": "energiess"}
EXPECTED_SHAPES = {"2d": (128, 128), "ra": (50, 128), "ac": (401,)}


def modality_dirs(root: Path) -> dict[str, Path] | None:
    """Return modality directories for a supported recording layout."""

    layouts = (
        {"2d": root / "mm" / "2d", "ra": root / "mm" / "ra", "ac": root / "ac"},
        {"2d": root / "-mm" / "2d", "ra": root / "-mm" / "ra", "ac": root / "-ac"},
    )
    return next(
        (
            layout
            for layout in layouts
            if all(directory.is_dir() for directory in layout.values())
        ),
        None,
    )


def file_map(directory: Path) -> dict[int, Path]:
    """Index integer-named MAT files by sample identifier."""

    result: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".mat":
            continue
        try:
            sample_id = int(path.stem)
        except ValueError as error:
            raise ValueError(f"MAT filename must be an integer: {path.name}") from error
        if sample_id in result:
            raise ValueError(f"duplicate sample id: {sample_id}")
        result[sample_id] = path

    if not result:
        raise FileNotFoundError(f"no MAT files found in {directory.name}")
    return result


def load_array(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    """Load and validate one signal array."""

    payload = loadmat(path)
    key = next((name for name in keys if name in payload), None)
    if key is None:
        available = [name for name in payload if not name.startswith("__")]
        if len(available) != 1:
            raise KeyError(
                f"{path.name}: expected {keys}, available variables={available}"
            )
        key = available[0]

    values = np.asarray(payload[key]).squeeze()
    if np.iscomplexobj(values):
        values = np.abs(values)
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values found in {path.name}")
    return values


def filter_mmwave(values: np.ndarray) -> np.ndarray:
    """Retain samples whose energy is near the frame peak."""

    energy = np.maximum(values - float(values.min()), 0.0) ** 2
    peak = float(energy.max())
    if peak <= 1e-12:
        return values
    return values * (energy >= peak * MMWAVE_PEAK_RATIO)


def filter_acoustic(values: np.ndarray) -> np.ndarray:
    """Retain the strongest CIR peak between its nearest local valleys."""

    values = values.reshape(-1)
    magnitude = np.abs(values)
    if float(magnitude.max()) <= 1e-12:
        return values

    peak = int(np.argmax(magnitude))
    left_limit = max(0, peak - CIR_PEAK_HALF_WINDOW)
    right_limit = min(values.size - 1, peak + CIR_PEAK_HALF_WINDOW)
    left_valley = left_limit
    right_valley = right_limit

    for index in range(peak - 1, left_limit, -1):
        if (
            magnitude[index] <= magnitude[index - 1]
            and magnitude[index] <= magnitude[index + 1]
        ):
            left_valley = index
            break

    for index in range(peak + 1, right_limit):
        if (
            magnitude[index] <= magnitude[index - 1]
            and magnitude[index] <= magnitude[index + 1]
        ):
            right_valley = index
            break

    start = min(left_valley + 1, peak)
    stop = max(right_valley, peak + 1)
    filtered = np.zeros_like(values)
    filtered[start:stop] = values[start:stop]
    return filtered


def save_mat(path: Path, key: str, values: np.ndarray) -> None:
    """Write one MAT file atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    savemat(
        temporary,
        {key: np.asarray(values, dtype=np.float32)},
        appendmat=False,
    )
    os.replace(temporary, path)


def process_dataset(source: Path, target: Path, force: bool) -> tuple[int, int]:
    """Preprocess one synchronized dataset."""

    directories = modality_dirs(source)
    if directories is None:
        raise FileNotFoundError("required modality directories were not found")

    maps = {name: file_map(path) for name, path in directories.items()}
    sample_sets = {name: set(paths) for name, paths in maps.items()}
    if not (sample_sets["2d"] == sample_sets["ra"] == sample_sets["ac"]):
        raise ValueError("modality sample identifiers are inconsistent")

    written = 0
    reused = 0
    for sample_id in sorted(sample_sets["2d"]):
        outputs = {
            "2d": target / "mm" / "2d" / f"{sample_id}.mat",
            "ra": target / "mm" / "ra" / f"{sample_id}.mat",
            "ac": target / "ac" / f"{sample_id}.mat",
        }
        if not force and all(path.is_file() for path in outputs.values()):
            reused += 1
            continue

        arrays = {
            name: load_array(maps[name][sample_id], MAT_KEYS[name])
            for name in maps
        }
        for name, expected in EXPECTED_SHAPES.items():
            if arrays[name].shape != expected:
                raise ValueError(
                    f"{name}/{sample_id}.mat has shape {arrays[name].shape}; "
                    f"expected {expected}"
                )

        arrays["2d"] = filter_mmwave(arrays["2d"])
        arrays["ra"] = filter_mmwave(arrays["ra"])
        arrays["ac"] = filter_acoustic(arrays["ac"]).reshape(-1, 1)
        for name in ("2d", "ra", "ac"):
            save_mat(outputs[name], OUTPUT_KEYS[name], arrays[name])
        written += 1

    return written, reused


def discover_datasets(input_root: Path) -> list[Path]:
    """Find supported datasets below an input root."""

    if modality_dirs(input_root) is not None:
        return [input_root]

    return sorted(
        (
            path
            for path in input_root.rglob("*")
            if path.is_dir() and modality_dirs(path) is not None
        ),
        key=lambda path: str(path.relative_to(input_root)).lower(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess synchronized mmWave and acoustic recordings."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite samples that already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError("input directory does not exist")
    if input_root == output_root:
        raise ValueError("input and output directories must be different")
    if input_root in output_root.parents:
        raise ValueError("output directory must not be inside the input directory")

    datasets = discover_datasets(input_root)
    if not datasets:
        raise FileNotFoundError("no supported dataset directories were found")

    for source in datasets:
        relative = Path(".") if source == input_root else source.relative_to(input_root)
        target = output_root if relative == Path(".") else output_root / relative
        written, reused = process_dataset(source, target, args.force)
        label = "." if relative == Path(".") else relative.as_posix()
        print(f"{label}: written={written}, reused={reused}")


if __name__ == "__main__":
    main()
