# Human-review break-even and override-log tests (supplementary, simulated)
- `reviewer_sensitivity.py SCENARIO [R]` then `agg_rs.py`: break-even FN:FP cost ratio w* for a dismissal-only reviewer over noise (sigma), automation bias and fatigue. Needs `run_experiment_prospective.py` on PYTHONPATH.
- `audit_log_test.py [R]`: override-log reconstruction ablations and tamper-detection test (hash chain, with and without an anchored head).
Both are design checks on synthetic data; they are not evidence about real reviewers.
