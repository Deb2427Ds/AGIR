# CORRECTED_GATE_TARGET_SPEC.md

## Previous (incorrect) target
```python
y = 0 if label == 'neutral' else 1
```
This maps beneficial->1, harmful->1, neutral->0 — measuring 'any non-neutral effect', not beneficial replay. Invalidates prior AUROC interpretations as 'beneficial-replay discrimination'.

## Corrected target
```python
y = 1 if label == 'beneficial' else 0
```
beneficial->1, harmful->0, neutral->0.
