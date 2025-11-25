
import requests, os, tqdm

URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
}

def ensure_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    for fname, url in URLS.items():
        dest = os.path.join(models_dir, fname)
        if os.path.exists(dest):
            print(f"[ok] {fname} ya existe")
            continue
        print(f"[+] Descargando {fname} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm.tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"[ok] Guardado en {dest}")

if __name__ == "__main__":
    ensure_models()
