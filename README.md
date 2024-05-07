## SadTalker-TensorRT

This project uses TensorRT to accelerate the [SadTalker](https://github.com/OpenTalker/SadTalker) facerender model.

#### What I did

- Refactor `src/facerender/modules/generator.py` to let `OcclusionAwareSPADEGenerator` accept tensor inputs other than tensor dict.
- Write script to export `AnimateFromCoeff` generator model to ONNX format and then convert to TensorRT Engine.
- Develop [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin) to support `grid_sample3d` operation in TensorRT.


#### Usage

1. Clone [grid-sample3d-trt-plugin](https://github.com/SeanWangJS/grid-sample3d-trt-plugin) repository and build it, then put the `.so` file into `./tensorrt/generator/plugin` directory.

2. Export Onnx model and create Engine.

```bash
cd tensorrt/generator
python export_onnx.py --checkpoint_dir /path/to/SadTalker-checkpoints
python create_engine.py
```

3. Run inference.

```bash
python inference_trt.py --checkpoint_dir /path/to/SadTalker-checkpoints
```