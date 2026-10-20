import os
import subprocess
from datetime import date
import requests
import pathlib

# Config
TNS_API_KEY = os.environ.get("TNS_ANIBOT_API")
TNS_USER_AGENT = os.environ.get("TNS_ANIBOT_USER_AGENT")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# MASTER_FILE = DATA_DIR / "all_objects.csv"
FILENAME = "tns_public_objects.csv.zip"
ZIP_FILE = os.path.join(DATA_DIR, FILENAME)
CSV_FILE = os.path.join(DATA_DIR, "tns_public_objects.csv")

URL = "https://www.wis-tns.org/system/files/tns_public_objects/tns_public_objects.csv.zip"


# download the file
with open(ZIP_FILE, "wb") as f:
    subprocess.run(
        ["curl", "-X", "POST",
         "-H", f"user-agent: {TNS_USER_AGENT}",
         "-d", f"api_key={TNS_API_KEY}",
         URL],
        stdout=f,
        check=True,
    )

# unzip the file
subprocess.run(["unzip", "-o", ZIP_FILE, "-d", DATA_DIR], check=True)

# remove the zip file
os.remove(ZIP_FILE)

print(f"Downloaded and extracted TNS public objects to {CSV_FILE}")