# Data placeholder

This repository does not redistribute the EGMSQA benchmark JSONL files or the large intermediate score artifacts used during development.

Scripts in this repository expect dataset files similar to:

- `dev_balanced.jsonl`
- `dev.jsonl`
- `test_balanced.jsonl`
- prompt JSONL files rendered from those datasets

At minimum, each example should expose:

- `id`
- `task_name`
- `question`
- `context.series`
- `context.static`
- `eval`

Place local copies of those files under `data/` or point the scripts to your own absolute paths.
