# AdaptLadder-BCI static website

This folder is a GitHub Pages-ready static site.

## Local preview
```bash
cd 02_GitHub_Pages_Static_Site
python3 -m http.server 8000
```
Open http://localhost:8000.

## Deploy to GitHub Pages
Option A - project repository `/docs` folder:
1. Copy all files from this folder into `docs/` in `https://github.com/ronnuriel/AdaptLadder-BCI`.
2. Commit and push.
3. In GitHub: Settings -> Pages -> Build and deployment -> Deploy from branch.
4. Select branch `main` and folder `/docs`.
5. After GitHub publishes, the site should be available as a project site under `https://ronnuriel.github.io/AdaptLadder-BCI/`.

Option B - `gh-pages` branch:
1. Create a `gh-pages` branch.
2. Put these files at the root of that branch.
3. Set Pages source to `gh-pages` / root.

The file `.nojekyll` is included so GitHub Pages serves static assets directly.
