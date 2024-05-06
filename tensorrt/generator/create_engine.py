import os
import ctypes

import torch
import tensorrt as trt

if __name__ == "__main__":

    engine_path = "./weights/model.engine"
    onnx_path = "./weights/model_gs.onnx"

    logger = trt.Logger(trt.Logger.VERBOSE)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    trt.init_libnvinfer_plugins(logger, "")
    handle=ctypes.CDLL("./plugin/libgrid_sample_3d_plugin.so", mode = ctypes.RTLD_GLOBAL)
    if not handle:
        print("load grid_sample_3d plugin error")

    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    config  = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    parser  = trt.OnnxParser(network, logger)

    config.max_workspace_size = 4 * 1<<30
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    # config.set_flag(trt.BuilderFlag.FP16)
    # config.builder_optimization_level = 5

    source_image = torch.load("./data/source_image.pt")
    kp_driving = torch.load("./data/kp_driving.pt")
    kp_source = torch.load("./data/kp_source.pt")
    kp_driving_jacobian = kp_driving["jacobian"] if "jacobian" in kp_driving else None
    kp_source_jacobian = kp_source["jacobian"] if "jacobian" in kp_source else None
    kp_driving = kp_driving["value"]
    kp_source = kp_source["value"]

    source_image = source_image[0:1, ...]
    kp_source = kp_source[0:1, ...]
    kp_driving = kp_driving[0:1, ...]

    batch_size=source_image.shape[0]
    profile.set_shape(
        "source_image",
        min = (1,) + source_image.shape[1:],
        opt = (2,) + source_image.shape[1:],
        max = (2,) + source_image.shape[1:]
    )

    profile.set_shape(
        "kp_driving",
        min = (1,) + kp_driving.shape[1:],
        opt = (2,) + kp_driving.shape[1:],
        max = (2,) + kp_driving.shape[1:]
    )

    profile.set_shape(
        "kp_source",
        min = (1,) + kp_source.shape[1:],
        opt = (2,) + kp_source.shape[1:],
        max = (2,) + kp_source.shape[1:]
    )

    profile.set_shape(
        "kp_driving_jacobian",
        min = (1, 15, 3, 3),
        opt = (2, 15, 3, 3),
        max = (2, 15, 3, 3)
    )

    profile.set_shape(
        "kp_source_jacobian",
        min = (1, 15, 3, 3),
        opt = (2, 15, 3, 3),
        max = (2, 15, 3, 3)
    )

    config.add_optimization_profile(profile)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    print(network.num_layers)
    engineString = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engineString)

    print(f"Engine was created to {engine_path} successfully!")