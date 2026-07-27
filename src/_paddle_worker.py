"""
Standalone PaddleOCR worker launched as subprocess.
Avoids CUDA symbol conflicts with PyTorch.
"""
import json, sys

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    img_path = sys.argv[1]
    page_num = int(sys.argv[2])
    out_path = sys.argv[3]

    # Mock albumentations.pytorch to prevent PaddleOCR's import chain
    # (ppocr → albumentations → torch) from loading PyTorch into this process.
    # PyTorch and PaddlePaddle CUDA DLLs conflict in the same process.
    import types
    sys.modules.setdefault('albumentations.pytorch', types.ModuleType('albumentations.pytorch'))

    import paddle
    from paddleocr import PaddleOCR

    # Use GPU if PaddlePaddle can find its bundled cuDNN
    use_gpu = paddle.is_compiled_with_cuda()
    if not use_gpu:
        paddle.device.set_device("cpu")

    ocr = PaddleOCR(lang="japan", use_angle_cls=True, use_gpu=use_gpu, show_log=False)

    img = Image.open(img_path)
    arr = np.array(img.convert("RGB"))[:, :, ::-1].copy()
    results = ocr.ocr(arr, cls=True)

    blocks = []
    if results and results[0]:
        for line in results[0]:
            pts = line[0]
            text = line[1][0]
            conf = line[1][1]
            if conf >= 0.3:
                x0 = int(min(p[0] for p in pts))
                y0 = int(min(p[1] for p in pts))
                x1 = int(max(p[0] for p in pts))
                y1 = int(max(p[1] for p in pts))
                blocks.append({"bbox": [x0, y0, x1, y1], "text": text})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False)
