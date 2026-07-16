import os
import re
import pathlib

from tidez.logger import get_logger
logger = get_logger("get_data")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DESIRT_DATA_DIR = pathlib.Path("/ocean/projects/phy250012p/shared/3DTS/DECAM/DESIRT/SEARCH/3DTS/")

ALL_CANDIDATES_FILE = DATA_DIR / "all_candidate_fits_files.txt"
MAX_DETECTIONS_FILE = DATA_DIR / "max_detections_candidate_fits_files.txt"


def find_candidate_fits_files():
    candidate_fits_files = list(DESIRT_DATA_DIR.glob("*/**/*.fits"))
    logger.info(f"Found {len(candidate_fits_files)} candidate FITS files.")

    with open(ALL_CANDIDATES_FILE, "w") as f:
        for fits_file in candidate_fits_files:
            f.write(str(fits_file) + "\n")

    logger.info(f"All candidate FITS file paths written to {ALL_CANDIDATES_FILE}")
    return candidate_fits_files


def parse_fits_filename(path):
    """
    Filenames look like: <NumberOfDetections>_<ObservationId>.fits
    Returns (n_detections, observation_id) or None if it doesn't match.
    """
    fname = os.path.basename(str(path))
    m = re.match(r'^(\d+)_(.+)\.fits$', fname)
    if not m:
        logger.warning(f"Could not parse filename, skipping: {path}")
        return None
    n_detections = int(m.group(1))
    observation_id = m.group(2)
    return n_detections, observation_id


def write_max_detections_per_observation(candidate_fits_files):
    """
    For each ObservationId, keep only the file with the max NumberOfDetections.
    """
    best_per_observation = {}  # observation_id -> (n_detections, path)

    for fits_file in candidate_fits_files:
        parsed = parse_fits_filename(fits_file)
        if parsed is None:
            continue
        n_detections, observation_id = parsed
        if (
            observation_id not in best_per_observation
            or n_detections > best_per_observation[observation_id][0]
        ):
            best_per_observation[observation_id] = (n_detections, fits_file)

    with open(MAX_DETECTIONS_FILE, "w") as f:
        for observation_id, (n_detections, fits_file) in best_per_observation.items():
            f.write(str(fits_file) + "\n")

    logger.info(f"Max-detections file paths written to {MAX_DETECTIONS_FILE}")
    return best_per_observation

def total_size_bytes(file_paths):
    total = 0
    for p in file_paths:
        try:
            total += os.path.getsize(p)
        except OSError as e:
            logger.warning(f"Could not stat {p}: {e}")
    return total

def main():
    candidate_fits_files = find_candidate_fits_files()
    best_per_observation = write_max_detections_per_observation(candidate_fits_files)

    logger.info(
        f"Read {len(candidate_fits_files)} total FITS files; "
        f"picked {len(best_per_observation)} files (one max-detections file per unique observation ID)."
    )
    total_bytes = total_size_bytes(candidate_fits_files)
    logger.info(f"Total size: {total_bytes / (1024**3):.2f} GB")    

if __name__ == "__main__":
    main()