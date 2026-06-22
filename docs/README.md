# AdaptLadder site hotfix

Fixes:
- Rebuilds the DriftGate simulator with robust vanilla JavaScript.
- Improves simulator layout and mobile behavior.
- Adds cache-busting query strings for CSS/JS.
- Changes the video player to default muted/no-bot mode and prefers `assets/adaptladder_silent_teaser.mp4`.

To apply locally:

```bash
cp -R docs/* /path/to/AdaptLadder-BCI/docs/
git add docs
git commit -m "Fix DriftGate simulator and mute explainer video"
git push origin main
```
