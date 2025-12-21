import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. TheBloke/xxx-GGUF")
    ap.add_argument("--filename", required=True, help="GGUF filename, e.g. model.Q4_K_M.gguf")
    ap.add_argument("--token", default=None, help="Hugging Face token (optional)")
    args = ap.parse_args()

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    local_path = hf_hub_download(
        repo_id=args.repo,
        filename=args.filename,
        token=args.token,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded to: {local_path}")


if __name__ == "__main__":
    main()
