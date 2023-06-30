"""Convert ONNX models to TensorRT engines.

Mostly scraped from https://github.com/onnx/onnx-tensorrt

Enhanced with the following features:
    - Support FP32/FP16/INT8 precision
    - Support serialization/deserialization of TensorRT engines to speed up
"""
from __future__ import print_function
import tensorrt as trt
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
import onnx
from onnx import helper as onnx_helper
import numpy as np
import six
from six import string_types
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p
def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        if isinstance(error_string, bytes):
            error_string = error_string.decode("utf-8")
        raise RuntimeError("cudaSetDevice: " + error_string)

def count_trailing_ones(vals):
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Binding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
            self.index  = engine.get_binding_index(self.name)
            if self.index == -1:
                raise IndexError("Binding name not found: %s" % self.name)
        else:
            self.index = idx_or_name
            self.name  = engine.get_binding_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.binding_is_input(self.index)


        dtype = engine.get_binding_dtype(self.index)
        dtype_map = {trt.DataType.FLOAT: np.float32,
                        trt.DataType.HALF:  np.float16,
                        trt.DataType.INT8:  np.int8,
                        trt.DataType.BOOL: bool}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32

        self.dtype = dtype_map[dtype]
        shape = engine.get_binding_shape(self.index)

        self.shape = tuple(shape)
        # Must allocate a buffer of size 1 for empty inputs / outputs
        if 0 in self.shape:
            self.empty = True
            # Save original shape to reshape output binding when execution is done
            self.empty_shape = self.shape
            self.shape = tuple([1])
        else:
            self.empty = False
        self._host_buf   = None
        self._device_buf = None
    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf
    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf
    def get_async(self, stream):
        src = self.device_buffer
        dst = self.host_buffer
        src.get_async(stream, dst)
        return dst

def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x

def check_input_validity(input_idx, input_array, input_binding):
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape    = tuple(input_array.shape)

    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()) :
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                            (input_idx, trt_shape, onnx_shape))

    # Check dtype
    if input_array.dtype != input_binding.dtype:
        #TRT does not support INT64, need to convert to INT32
        if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
            casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
            if np.equal(input_array, casted_input_array).all():
                input_array = casted_input_array
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                            (input_idx, input_binding.dtype, input_array.dtype))
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    return input_array


class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine
        nbinding = self.engine.num_bindings

        bindings = [Binding(self.engine, i)
                    for i in range(nbinding)]
        self.binding_addrs = [b.device_buffer.ptr for b in bindings]
        self.inputs  = [b for b in bindings if     b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]

        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer   # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()

    def __del__(self):
        if self.engine is not None:
            del self.engine

    def run(self, inputs):
        # len(inputs) > len(self.inputs) with Shape operator, input is never used
        # len(inputs) == len(self.inputs) for other operators
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]


        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array = check_input_validity(i, input_array, input_binding)
            input_binding_array = input_binding.device_buffer
            input_binding_array.set_async(input_array, self.stream)

        self.context.execute_async_v2(
            self.binding_addrs, self.stream.handle)

        results = [output.get_async(self.stream)
                   for output in self.outputs]

        # For any empty bindings, update the result shape to the expected empty shape
        for i, (output_array, output_binding) in enumerate(zip(results, self.outputs)):
            if output_binding.empty:
                results[i] = np.empty(shape=output_binding.empty_shape, dtype=output_binding.dtype)

        self.stream.synchronize()
        return results

    def run_no_dma(self, batch_size):
        self.context.execute_async(
            batch_size, self.binding_addrs, self.stream.handle)

class TensorRTBackendRep(BackendRep):
    def __init__(self, model, device,
            max_workspace_size=None, serialize_engine=False, verbose=False,
            serialized_engine_path=None, **kwargs):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self.serialized_engine_path = serialized_engine_path
        self._logger = TRT_LOGGER
        self.builder = trt.Builder(self._logger)
        self.config = self.builder.create_builder_config()
        if self.builder.platform_has_fast_fp16:
            print("FAST FP16 detected. Enabling precision to FP16...")
            self.config.set_flag(trt.BuilderFlag.FP16)
#         if self.builder.platform_has_fast_int8:
#             print("FAST INT8 detected. Enabling INT8...")
#             self.config.set_flag(trt.BuilderFlag.INT8)
        self.network = self.builder.create_network(flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self._logger)
        self.shape_tensor_inputs = []
        self.serialize_engine = serialize_engine
        self.verbose = verbose

        if self.verbose:
            print(f'\nRunning {model.graph.name}...')
            TRT_LOGGER.min_severity = trt.Logger.VERBOSE

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)
        
        if max_workspace_size is None:
            max_workspace_size = 1 << 28

        self.config.max_workspace_size = max_workspace_size

        if self.verbose:
            for layer in self.network:
                print(layer)

            print(f'Output shape: {self.network[-1].get_output(0).shape}')

        if os.path.exists(self.serialized_engine_path):
            del self.parser
            self.runtime = trt.Runtime(TRT_LOGGER)
            print("Loading serialized engine from {}".format(self.serialized_engine_path))
            with open(self.serialized_engine_path, 'rb') as f:
                trt_engine = self.runtime.deserialize_cuda_engine(f.read())
            self.engine = Engine(trt_engine)
        else:
            self._build_engine()
        
        self._output_shapes = {}
        self._output_dtype = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type

    def _build_engine(self, inputs=None):
        """
        Builds a TensorRT engine with a builder config.
        :param inputs: inputs to the model; if not None, this means we are building the engine at run time,
                       because we need to register optimization profiles for some inputs
        :type inputs: List of np.ndarray
        """

        if inputs:
            opt_profile = self.builder.create_optimization_profile()
            # Set optimization profiles for the input bindings that need them
            for i in range(self.network.num_inputs):
                inp_tensor = self.network.get_input(i)
                name = inp_tensor.name
                # Set profiles for shape tensors
                if inp_tensor.is_shape_tensor:
                    if inputs[i].ndim > 0:
                        val_list = inputs[i].tolist()
                        opt_profile.set_shape_input(name, val_list, val_list, val_list)
                    else:
                        opt_profile.set_shape_input(name, [inputs[i]], [inputs[i]], [inputs[i]])
                # Set profiles for dynamic execution tensors
                elif -1 in inp_tensor.shape:
                    opt_profile.set_shape(name, inputs[i].shape, inputs[i].shape, inputs[i].shape)

            self.config.add_optimization_profile(opt_profile)

        trt_engine = self.builder.build_engine(self.network, self.config)

        if trt_engine is None:
            raise RuntimeError("Failed to build TensorRT engine from network")
        if self.serialize_engine:
            trt_engine = self._serialize_deserialize(trt_engine, self.serialized_engine_path)
        self.engine = Engine(trt_engine)

    def _set_device(self, device):
        self.device = device
        assert(device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)

    def _serialize_deserialize(self, trt_engine, serialized_engine_path):
        # TODO(roger): unify load and save functions
        self.runtime = trt.Runtime(TRT_LOGGER)
        serialized_engine = trt_engine.serialize()
        if serialized_engine_path is not None:
            with open(serialized_engine_path, 'wb') as f:
                f.write(serialized_engine)
            print("Serialized engine written to {}".format(serialized_engine_path))
        del self.parser # Parser no longer needed for ownership of plugins
        trt_engine = self.runtime.deserialize_cuda_engine(
                serialized_engine)
        return trt_engine

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        outputs = self.engine.run(inputs)
        output_names = [output.name for output in self.engine.outputs]

        for i, (name, array) in enumerate(zip(output_names, outputs)):
            output_shape = self._output_shapes[name]
            # HACK WAR for unknown output shape in run_node
            if output_shape == (-99,):
                # WAR for TRT requiring at least 2 dims (NC)
                min_dims = 2
                if _tensorrt_version()[0] < 4:
                    # WAR for TRT only supporting 4D (NCHW) tensors
                    min_dims = 4
                if array.ndim == min_dims:
                    npadding_dims = count_trailing_ones(array.shape)
                    if npadding_dims > 0:
                        outputs[i] = array.reshape(
                            array.shape[:-npadding_dims])
            else:
                # HACK WAR replace fixed batch dim with variable
                if self._output_dtype[name] == onnx.TensorProto.INT64 and array.dtype == np.int32:
                    casted_output = np.array(outputs[i], dtype=np.int64)
                    if np.equal(outputs[i], casted_output).all():
                        outputs[i] = np.array(outputs[i], dtype=np.int64)
                if self._output_dtype[name] == onnx.TensorProto.DOUBLE and array.dtype == np.float32:
                    casted_output = np.array(outputs[i], dtype=np.double)
                    if np.equal(outputs[i], casted_output).all():
                        outputs[i] = np.array(outputs[i], dtype=np.double)

        outputs_tuple = namedtupledict('Outputs', output_names)(*outputs)
        return namedtupledict('Outputs', output_names)(*outputs)

def np2onnx_dtype(np_dtype):
    if np_dtype == np.dtype('float32'):
        return onnx.TensorProto.FLOAT
    elif np_dtype == np.dtype('float16'):
        return onnx.TensorProto.FLOAT16
    elif np_dtype == np.dtype('int64'):
        return onnx.TensorProto.INT64
    elif np_dtype == np.dtype('int32'):
        return onnx.TensorProto.INT32
    elif np_dtype == np.dtype('int8'):
        return onnx.TensorProto.INT8
    elif np_dtype == np.dtype('double'):
        return onnx.TensorProto.DOUBLE
    else:
        raise TypeError("Unsupported data type:", np_dtype)

def make_node_test_model(node, inputs, use_weights=True):
    # HACK TODO: The output info is unknown here; not sure what the best solution is
    output_dtype = np.float32 # Dummy value only
    output_shape = [-99]      # Dummy value only
    graph_inputs = [onnx_helper.make_tensor_value_info(
        name, np2onnx_dtype(array.dtype), array.shape)
                    for name, array in zip(node.input, inputs)]
    graph_outputs = [onnx_helper.make_tensor_value_info(
        name, np2onnx_dtype(output_dtype), output_shape)
                     for name in node.output]
    if use_weights:
        # Add initializers for all inputs except the first
        initializers = [onnx_helper.make_tensor(
            name, np2onnx_dtype(array.dtype), array.shape, array.flatten().tolist())
                        for name, array in zip(node.input[1:], inputs[1:])]
    else:
        initializers = []
    graph = onnx_helper.make_graph(
           [node], "RunNodeGraph_" + node.op_type,
           graph_inputs, graph_outputs, initializer=initializers)
    model = onnx_helper.make_model(graph)
    return model

class TensorRTBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CUDA:0', **kwargs):
        """Build an engine from the given model.
        model -- An ONNX model as a deserialized protobuf, or a string or file-
                 object containing a serialized protobuf.
        """
        super(TensorRTBackend, cls).prepare(model, device, **kwargs)
        return TensorRTBackendRep(model, device, **kwargs)
    
    @classmethod
    def supports_device(cls, device_str):
        device = Device(device_str)
        return device.type == DeviceType.CUDA
