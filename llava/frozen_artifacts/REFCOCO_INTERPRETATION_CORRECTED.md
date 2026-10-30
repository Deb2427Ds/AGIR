# REFCOCO_INTERPRETATION_CORRECTED.md

## Superseded claim (too strong, do not use)

> "Grounding quality itself is not the bottleneck on LLaVA."

## Corrected interpretation

Grounding DINO demonstrates strong localization capability on RefCOCOg when supplied with the true referring expression (N=400, mIoU=0.7158, Acc@0.5=75.75%, pointing-game=81.25%, DINO no-box rate=0.5%). This isolates the grounding model's localization capacity in isolation, but does not by itself establish that LLaVA-generated evidence phrases (used as the DINO query in the main AGIR pipeline) are equally reliable on A-OKVQA. The RefCOCOg result should not be read as evidence that grounding quality is not the bottleneck in the main experiment.

## Underlying data (unchanged, no rerun performed)

N=400, mIoU=0.7158, Acc@0.5=75.75%, pointing-game=81.25%, DINO no-box rate=0.5%
