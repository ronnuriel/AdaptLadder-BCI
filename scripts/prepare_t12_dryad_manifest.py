from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path


DATASET_DOI = "doi:10.5061/dryad.x69p8czpq"
DATASET_API = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.x69p8czpq"


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def absolute_api_url(path_or_url: str) -> str:
    if path_or_url.startswith("http"):
        return path_or_url
    return f"https://datadryad.org{path_or_url}"


def write_manifest(output_csv: Path, output_notes: Path) -> None:
    dataset = fetch_json(DATASET_API)
    version_url = absolute_api_url(dataset["_links"]["stash:version"]["href"])
    version = fetch_json(version_url)
    files_url = absolute_api_url(version["_links"]["stash:files"]["href"])
    files = fetch_json(files_url)["_embedded"]["stash:files"]

    rows = []
    for item in files:
        file_id = item["_links"]["self"]["href"].rstrip("/").split("/")[-1]
        download_href = absolute_api_url(item["_links"]["stash:download"]["href"])
        rows.append(
            {
                "file_id": file_id,
                "path": item["path"],
                "size_bytes": item["size"],
                "size_gb": round(item["size"] / (1024**3), 4),
                "mime_type": item["mimeType"],
                "sha256": item["digest"],
                "download_api_url": download_href,
            }
        )
    rows = sorted(rows, key=lambda row: row["path"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    diagnostic = next(row for row in rows if row["path"] == "diagnosticBlocks.tar.gz")
    output_notes.write_text(
        "\n".join(
            [
                "# T12 Dryad Download Notes",
                "",
                f"Dataset: {DATASET_DOI}",
                f"Title: {version['title']}",
                f"Publication date: {version.get('publicationDate', '')}",
                f"Last modification date: {version.get('lastModificationDate', '')}",
                "",
                "This project uses T12 only as an optional feasibility validation.",
                "Start with `diagnosticBlocks.tar.gz`; avoid the much larger language-model and derived archives unless decoder reproduction becomes necessary.",
                "",
                "Suggested local layout after manual download/extract:",
                "",
                "```text",
                "data/raw/t12_diagnosticBlocks/",
                "  *.mat",
                "```",
                "",
                "DiagnosticBlocks file:",
                f"- Dryad file id: {diagnostic['file_id']}",
                f"- Size: {diagnostic['size_gb']} GB",
                f"- SHA-256: {diagnostic['sha256']}",
                f"- API download URL: {diagnostic['download_api_url']}",
                "",
                "If direct API download requires browser/session authorization, use the Dryad web page and keep the archive out of git.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the Dryad file manifest for the T12 speech neuroprosthesis dataset.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/external/t12_high_performance_speech/dryad_file_manifest.csv"),
    )
    parser.add_argument(
        "--output-notes",
        type=Path,
        default=Path("data/external/t12_high_performance_speech/download_notes.md"),
    )
    args = parser.parse_args()
    write_manifest(args.output_csv, args.output_notes)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_notes}")


if __name__ == "__main__":
    main()
