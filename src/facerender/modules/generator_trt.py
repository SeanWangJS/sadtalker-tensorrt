from typing import Dict, Union
import ctypes

from cuda import cudart
import tensorrt as trt
import numpy as np
import torch

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        trt.IOutputAllocator.__init__(self)
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        ptr = cudart.cudaMalloc(size)[1]
        self.buffers[tensor_name] = ptr
        return ptr
    
    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)

def load_plugin(plugin_path: str, logger: trt.Logger):
    success = ctypes.CDLL(plugin_path, mode = ctypes.RTLD_GLOBAL)
    if not success:
        print("load grid_sample_3d plugin error")
        raise Exception()

    trt.init_libnvinfer_plugins(logger, "")

def load_engine(engine_path: str, logger: trt.Logger):
    with open(engine_path, "rb") as f:
        engineString = f.read()
    
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()
    return engine, context

def inference(engine, context, inputs: dict):

    ## setup input
    input_buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        
        array = inputs[name]
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        array = array.astype(dtype)
        array = np.ascontiguousarray(array)

        err, ptr = cudart.cudaMalloc(array.nbytes)
        if err > 0:
            raise Exception("cudaMalloc failed, error code: {}".format(err))
        input_buffers[name] = ptr
        cudart.cudaMemcpy(ptr, array.ctypes.data, array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        context.set_input_shape(name, array.shape)
        context.set_tensor_address(name, ptr)

    ## setup output
    output_allocator = OutputAllocator()
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue

        context.set_output_allocator(name, output_allocator)
    
    ## execute
    context.execute_async_v3(0)

    ## fetch output
    output = {}
    for name in output_allocator.buffers.keys():
        ptr = output_allocator.buffers[name]
        shape = output_allocator.shapes[name]
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        nbytes = np.prod(shape) * dtype.itemsize
        
        output_buffer = np.empty(shape, dtype = dtype)
        cudart.cudaMemcpy(output_buffer.ctypes.data, ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        output[name] = output_buffer
    
    ## free input buffers
    for name in input_buffers.keys():
        ptr = input_buffers[name]
        cudart.cudaFree(ptr)
    
    ## free output buffers
    for name in output_allocator.buffers.keys():
        ptr = output_allocator.buffers[name]
        cudart.cudaFree(ptr)

    return output

class OcclusionAwareSPADEGenerator:
    def __init__(self, engine_path: str, plugin_path: str):
        logger  = trt.Logger(trt.Logger.VERBOSE)
        load_plugin(plugin_path, logger)
        self.engine, self.context = load_engine(engine_path, logger)
        
    def __call__(self, 
                 source_image: Union[torch.Tensor, np.ndarray], 
                 kp_driving: Dict[str, Union[torch.Tensor, np.ndarray]], 
                 kp_source: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, torch.Tensor]:
        
        kp_driving_jacobian = kp_driving["jacobian"] if "jacobian" in kp_driving else None
        kp_source_jacobian = kp_source["jacobian"] if "jacobian" in kp_source else None
        kp_driving = kp_driving["value"]
        kp_source = kp_source["value"]

        if isinstance(source_image, torch.Tensor):
            source_image = source_image.detach().cpu().numpy()
        if isinstance(kp_driving, torch.Tensor):
            kp_driving = kp_driving.detach().cpu().numpy()
        if isinstance(kp_source, torch.Tensor):
            kp_source = kp_source.detach().cpu().numpy()
        if isinstance(kp_driving_jacobian, torch.Tensor):
            kp_driving_jacobian = kp_driving_jacobian.detach().cpu().numpy()
        if isinstance(kp_source_jacobian, torch.Tensor):
            kp_source_jacobian = kp_source_jacobian.detach().cpu().numpy()

        inputs = {
            "source_image": source_image,
            "kp_driving": kp_driving,
            "kp_source": kp_source,
            "kp_driving_jacobian": kp_driving_jacobian,
            "kp_source_jacobian": kp_source_jacobian
        }

        """
        {
            "mask": mask,
            "occlusion_map": occlusion_map,
            "prediction": prediction 
        }
        """
        output = inference(self.engine, self.context, inputs)
        return output