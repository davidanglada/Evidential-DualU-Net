# Legacy entry points

The original checkpoint-compatible entry points remain at repository root:

- `train.py`
- `eval.py`
- `eval_unc.py`
- `unc_study_cent.py`

They are intentionally left in place so existing cluster commands and paper experiments do not break. New users should use the validated wrappers one directory above. The large centroid uncertainty study is experimental and is not part of the minimal public workflow.

