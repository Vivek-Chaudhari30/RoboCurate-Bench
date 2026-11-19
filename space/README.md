---
title: RoboCurate-Bench
emoji: 🧹
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Placebo-controlled benchmark for robot-data cleaners
tags:
  - robotics
  - lerobot
  - benchmark
  - data-curation
---

# RoboCurate-Bench leaderboard (static Space)

This directory is a Hugging Face **static** Space. The front-matter keys are from the Spaces configuration reference
(huggingface/hub-docs@37ae8ca8, `docs/hub/spaces-config-reference.md`: `sdk` is `gradio`, `docker` or `static`;
`app_file` is the main HTML file; colours are red, yellow, green, blue, indigo, purple, pink or gray), and
`docs/hub/spaces-sdks-static.md` (static Spaces serve `index.html`).

Regenerate the page from a results directory:

```bash
robocurate report <results_dir> --site space/
```

`index.html` and `leaderboard.json` are overwritten; this README is kept. FAKE (synthetic-backend) results are
refused here. This Space has not been deployed; deploying is a manual step by the maintainer.
