# T12 Dryad Download Notes

Dataset: doi:10.5061/dryad.x69p8czpq
Title: Data for: A high-performance speech neuroprosthesis
Publication date: 2023-09-01
Last modification date: 2023-09-01

This project uses T12 only as an optional feasibility validation.
Start with `diagnosticBlocks.tar.gz`; avoid the much larger language-model and derived archives unless decoder reproduction becomes necessary.

Suggested local layout after manual download/extract:

```text
data/raw/t12_diagnosticBlocks/
  *.mat
```

DiagnosticBlocks file:
- Dryad file id: 2547371
- Size: 0.5703 GB
- SHA-256: dc204dda6a67f25824f6f9b7a7cb6b83ca8142376a9787ebb66fa223402882e7
- API download URL: https://datadryad.org/api/v2/files/2547371/download

If direct API download requires browser/session authorization, use the Dryad web page and keep the archive out of git.
