"""
Step 4b: download the TNS public-objects dump to data/tns_public_objects.csv.

basic_filtering.py reads it to flag objects already on TNS and to tell whether
our DECam objid is among the TNS internal names.

Needs TNS_ANIBOT_API and TNS_ANIBOT_USER_AGENT in the environment.

Usage:
    python tidez/download_tns_objects.py
"""

import os
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_FILE = DATA_DIR / "tns_public_objects.csv.zip"
CSV_FILE = DATA_DIR / "tns_public_objects.csv"

URL = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"


def main() -> None:
    api_key = os.environ.get("TNS_ANIBOT_API")
    user_agent = os.environ.get("TNS_ANIBOT_USER_AGENT")
    if not api_key or not user_agent:
        sys.exit("TNS_ANIBOT_API / TNS_ANIBOT_USER_AGENT are not set; export them first.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(ZIP_FILE, "wb") as f:
        subprocess.run(["curl", "-sS", "--fail", "-X", "POST",
                        "-H", f"user-agent: {user_agent}",
                        "-d", f"api_key={api_key}",
                        URL],
                       stdout=f, check=True)
    subprocess.run(["unzip", "-o", str(ZIP_FILE), "-d", str(DATA_DIR)], check=True)
    ZIP_FILE.unlink()
    print(f"Downloaded and extracted TNS public objects to {CSV_FILE}")


if __name__ == "__main__":
    main()
