import argparse, yaml, pathlib
def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # ensure dirs
    pathlib.Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg

