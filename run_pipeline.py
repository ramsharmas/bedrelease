"""One command: verify data -> train -> test -> write outputs/.   Usage: python run_pipeline.py [--download]"""
import sys
from bedrelease import data, pipeline

if __name__ == "__main__":
    if "--download" in sys.argv:
        print("Downloading data from the pinned GitHub mirror ...")
        print("saved:", data.download_raw(force=True))
    pipeline.run()

# 90%+ mode: model on SYNTHETIC in-stay signals (clearly labelled simulated)
if __name__ == "__main__" and "--skip-synthetic" not in sys.argv:
    import runpy; print("[extra] training synthetic in-stay model ..."); runpy.run_path("scripts/train_synthetic.py")
