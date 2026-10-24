# Pretrained weights

Place the AI-KD DifFIQA(R) face image quality weights here:

```
pretrained/aikd_diffiqar_model.pth
```

`train.py` looks for this file by default (`--fiqa_weights`). It is needed by
the `full` and `wo_gen` variants only. Weight files are git-ignored.
