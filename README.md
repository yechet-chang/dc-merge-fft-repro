# FFT Reproduction Reference

This repository is a minimal FFT-only reproduction reference for the `DTD` / `EuroSAT` mismatch.

Included:

- evaluation code
  - `eval_single_task.py`
  - `src/datasets/emnist.py`
  - `src/models/task_vectors.py`
- one minimal result summary
  - `results/single_task/ViT-B-32/RUN_AVERAGES_20260413.md`

Large local assets such as datasets and checkpoints are intentionally not included in this repository.

Run from this directory with:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_single_task.py --model=ViT-B-32 --finetuning-mode=none --model-location=./checkpoints/fft/checkpoints --data-location=./datasets
```

and:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard --model-location=./checkpoints/fft/checkpoints --data-location=./datasets
```

Current observation:

- 8-task zero-shot average: `0.4812568308433799`
- 8-task individual average: `0.9290763888003861`
- 8-task baseline average: `0.9282918143323010`
- main remaining mismatches:
  - `DTD: 0.976063829787 vs 0.979787234043`
  - `EuroSAT: 0.999259259259 vs 0.989259259259`

## Acknowledgements

This minimal reproduction package is based on the original repositories:

- `DC-Merge`: <https://github.com/EnnengYang/DC-Merge>
- `task_vectors`: <https://github.com/mlfoundations/task_vectors>

The reference single-task baseline file is derived from the released FFT results in the original project context.
