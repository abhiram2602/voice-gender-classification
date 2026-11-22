"""Download and validate public voice dataset; CC BY-NC-SA 4.0."""
import csv
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/primaryobjects/voice-gender/master/voice.csv"

def download(path="data/voice.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 3168 or len(reader.fieldnames) != 21 or "label" not in reader.fieldnames:
        path.unlink(missing_ok=True)
        raise ValueError("Unexpected voice dataset schema")
    print(f"Validated {len(rows)} rows: {path}")

if __name__ == "__main__":
    download()
