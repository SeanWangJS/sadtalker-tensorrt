import sys
import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)
print(PROJECT_DIR)
import argparse

import torch
import numpy as np
import onnx
import onnx_graphsurgeon as gs

from src.utils.init_path import init_path
from src.facerender.animate import AnimateFromCoeff

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## The following arguments are taked from ../../inference.py file
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the directory containing the model weigths")
    parser.add_argument("--size", type=int, default=256, help="The image size of the facerender")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="How to preprocess the images" ) 
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(PROJECT_DIR, 'src/config'), args.size, preprocess = args.preprocess)

    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    model = animate_from_coeff.generator

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
    with torch.no_grad():
        symbolic_names = {0: "batch_size"}
        torch.onnx.export(model=model,
            args = (source_image, kp_driving, kp_source, kp_driving_jacobian, kp_source_jacobian),
            f = "./weights/model.onnx",
            opset_version=16,
            do_constant_folding=True, 
            input_names=["source_image", "kp_driving", "kp_source", "kp_driving_jacobian", "kp_source_jacobian"],
            output_names=["mask", "occlusion_map", "prediction"],
            dynamic_axes={
                "source_image": symbolic_names,
                "kp_driving": symbolic_names,
                "kp_source": symbolic_names
            }
        )

    model = onnx.load("./weights/model.onnx")
    graph = gs.import_onnx(model)
    i = 0
    for node in graph.nodes:
        i = i + 1
        if "GridSample" in node.name:
            input = node.inputs[0]
            grid = node.inputs[1]
            output=node.outputs[0]
            node.attrs = {"name": "GridSample3D", "version": 1, "namespace": ""}
            node.op = "GridSample3D"
            
    onnx.save(gs.export_onnx(graph), "./weights/model_gs.onnx")
    print("Save model to ./weights/model_gs.onnx")