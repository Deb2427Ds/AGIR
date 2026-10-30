# Random budget-matched gate — methods (Task C)

10,000 draws without replacement of exactly 635/1,145 validation IDs, numpy Generator(PCG64) seeded 42, drawn once and not selected for favorable outcome. Per-draw soft accuracy = accuracy_dino for replayed IDs, accuracy_direct otherwise (matches the manuscript's soft-score convention; no per-sample binarization). Beneficial/harmful/neutral labels are the fixed sign of (accuracy_dino - accuracy_direct), identical to the manuscript's label rule, computed once from already-frozen scores. No model call, no retraining, no threshold search.
