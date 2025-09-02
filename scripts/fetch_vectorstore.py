#!/usr/bin/env python3
"""Small helper to ensure vectorstore files are present.

Intended to be run during image build or as a startup step:

  python -m scripts.fetch_vectorstore --repo-id SamPease/TransAdviceAgent

It calls `ensure_vectorstore_files()` from `app.rag_pipeline` which already
handles HF_TOKEN for private datasets and will skip existing files.
"""
import argparse
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="SamPease/TransAdviceAgent", help="HuggingFace repo id (dataset or repo)")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"], help="HF repo type")
    parser.add_argument("--files", nargs="*", help="Specific files to ensure (defaults to index.faiss, docs.sqlite, id_map.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("fetch_vectorstore")

    try:
        # local import to avoid top-level failures if app package not yet on PYTHONPATH
        from app.rag_pipeline import ensure_vectorstore_files

        log.info("Ensuring vectorstore files from %s (type=%s)", args.repo_id, args.repo_type)
        ensure_vectorstore_files(repo_id=args.repo_id, files=args.files, repo_type=args.repo_type)
        log.info("Done ensuring vectorstore files")
    except Exception:
        log.exception("Failed to ensure vectorstore files")
        raise


if __name__ == "__main__":
    main()
