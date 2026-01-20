import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from config import data_raw_dir


TARGET_SIZE = (256, 256)
VALID_SUFFIXES = {".png", ".jpg", ".jpeg"}

try:  # Pillow>=9.1 provides Image.Resampling
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # Fallback for older Pillow versions
    RESAMPLE = Image.LANCZOS


def _resize_one(file_path: str) -> tuple[str, bool, str]:
    """Resize a single image in-place and report status."""
    path = Path(file_path)
    try:
        with Image.open(path) as img:
            if img.size == TARGET_SIZE:
                return path.name, False, ""
            resized = img.resize(TARGET_SIZE, RESAMPLE)
            resized.save(path)
        return path.name, True, ""
    except Exception as exc:  # Surface problematic files without stopping the batch
        return path.name, False, str(exc)


def main() -> None:
    data_dir = Path(data_raw_dir) / "GastroNet-spaarne"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_paths = [
        str(path)
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]

    if not image_paths:
        print(f"No PNG/JPG images found in {data_dir}.")
        return

    resized = skipped = failed = 0
    max_workers = min(32, os.cpu_count() or 1)

    # Process images in parallel to keep large batches fast.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for name, changed, error in tqdm(
            executor.map(_resize_one, image_paths),
            total=len(image_paths),
            desc="Resizing images",
        ):
            if error:
                failed += 1
                tqdm.write(f"Failed: {name} ({error})")
            elif changed:
                resized += 1
            else:
                skipped += 1

    print(
        f"Done. Resized: {resized}, already correct size: {skipped}, failures: {failed}."
    )


if __name__ == "__main__":
    main()