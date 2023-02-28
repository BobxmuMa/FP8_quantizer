import torch, math, struct
from core import (PPQ_CONFIG, QuantizationProperty, QuantizationStates,
                      RoundingPolicy, TensorQuantizationConfig)
from torch.autograd import Function

ROUND_HALF_EVEN          = 0
ROUND_HALF_UP            = 1
ROUND_HALF_DOWN          = 2
ROUND_HALF_TOWARDS_ZERO  = 3
ROUND_HALF_FAR_FORM_ZERO = 4
ROUND_TO_NEAR_INT        = 5
ROUND_UP                 = 6
ROUND_DOWN               = 7

def CheckTensor(tensor, dtype, name):
    if tensor.dtype != dtype:
        raise ValueError("Kernel Failure, Invalid dtype of Input tensor: " + name)
    if tensor == torch.Size([]):
        raise ValueError("Kernel Failure, Tensor is empty: " + name)
    

class TensorwiseFloatingQuantImpl(Function):
    """Torch Tensorwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will use ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor,
                exponet_bits: int, mantissa_bits: int,
                quant_min: float, quant_max: float,
                rounding: RoundingPolicy) -> torch.Tensor:

        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            # quantization function, pytorch implmentation
            raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
        
        else:
            from core import CUDA

            # quantization function, pure cuda implmentation
            quantized = CUDA.FloatingQuantize_T(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                exponent=exponet_bits,
                mantissa=mantissa_bits,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value
            )
            return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None


def _round2int(value, rounding):
    if rounding == ROUND_HALF_EVEN:
        return torch.round(value) if torch.is_tensor(value) else round(value)
    elif rounding == ROUND_HALF_UP:
        return (value + .5).floor()
    elif rounding == ROUND_HALF_DOWN:
        return (value - .5).ceil()
    elif ROUND_HALF_TOWARDS_ZERO:
        if value > 0: return _round2int(value, ROUND_HALF_DOWN)
        else: return _round2int(value, ROUND_HALF_UP)
    elif ROUND_HALF_FAR_FORM_ZERO:
        if value > 0: return _round2int(value, ROUND_HALF_UP)
        else: return _round2int(value, ROUND_HALF_DOWN)
    elif ROUND_UP:
        value.ceil()
    elif ROUND_DOWN:
        value.floor()
    else:
        return torch.round(value)

def float_to_uint(num):
    return struct.unpack('!I', struct.pack('!f', num))[0]

def uint_to_float(num):
    return struct.unpack('!f', struct.pack('!I', num))[0]

def QuantizeScalarFloating(
    value, scale, offset, exponent, mantissa,
    clip_min, clip_max, rounding):
    helper, rounding_helper = {"value":None, "data":None}, {"value":None, "data":None}
    Unscaled_FP32 = value / scale
    

    helper["value"] = Unscaled_FP32
    helper["data"] = float_to_uint(Unscaled_FP32)
    exponent_min  = -(1 << (exponent - 1)) + 1
    exponent_max  = (1 << (exponent - 1))
    # print("helper[value]: {} helper[data]: {}, exponent_min: {}, exponent_max: {}".format\
    #     (helper["value"], float_to_uint(helper["data"]) , exponent_min, exponent_max))

    fp32_sign     = 0
    fp32_exp      = (exponent_max + 127) << 23
    fp32_mantissa = ~(0x007FFFFF >> mantissa) & 0x007FFFFF
    helper["value"] = uint_to_float(fp32_sign + fp32_mantissa + fp32_exp)
    helper["data"] = fp32_sign + fp32_mantissa + fp32_exp
    theoretical_maximum = helper["value"]
    # print("fp32_sign: {} fp32_exp: {}, fp32_mantissa: {}, helper[data]: {}, theoretical_maximum: {}".format\
    #     (fp32_sign, fp32_exp, fp32_mantissa, helper["data"], theoretical_maximum))

    if (Unscaled_FP32 > min(clip_max, theoretical_maximum)):
        return min(clip_max, theoretical_maximum)
    if (Unscaled_FP32 < max(clip_min, -theoretical_maximum)):
        return max(clip_min, -theoretical_maximum)

    helper["value"] = Unscaled_FP32
    helper["data"] = float_to_uint(Unscaled_FP32)
    # print(helper["data"], 0x80000000)
    # print(helper["data"].dtype, helper["data"].int().dtype)
    # import sys; sys.exit(1)
    fp32_sign     = helper["data"] & 0x80000000
    fp32_exp      = helper["data"] & 0x7F800000
    fp32_mantissa = helper["data"] & 0x007FFFFF
    
    if (((fp32_exp >> 23) - 127) < exponent_min + 1):
        min_subnormal = float(1.0) / (1 << ((1 << (exponent - 1)) + mantissa - 2))
        return _round2int(Unscaled_FP32 / min_subnormal, rounding) * min_subnormal
    
    rounding_helper["data"] = ((fp32_mantissa << (mantissa)) & 0x007FFFFF) + 0x3F800000
    rounding_helper["value"] = uint_to_float(((fp32_mantissa << (mantissa)) & 0x007FFFFF) + 0x3F800000)
    round_bit = _round2int(rounding_helper["value"] - 1, rounding)

    fp32_mantissa = ((fp32_mantissa >> (23 - mantissa)) + round_bit) << (23 - mantissa)
    helper["data"] = fp32_sign + fp32_mantissa + fp32_exp
    helper["value"] = uint_to_float(fp32_sign + fp32_mantissa + fp32_exp)

    return torch.clamp(torch.tensor(helper["value"]), clip_min, clip_max)

def DequantizeScalar(value, scale, offset):
    return (value - offset) * scale

def QuantizeTensor_FC(num_of_element, element_per_channel, num_of_channel,
    value, scale, offset, exponent, mantissa, clip_min, clip_max, rounding):
    quantized_value = torch.empty_like(value)
    value_size = value.size()
    value = value.reshape(-1)
    quantized_value = quantized_value.reshape(-1)
    for i in range(num_of_element):
        c = int(i / element_per_channel) % num_of_channel

        qt = QuantizeScalarFloating(
        value[i], scale[c], offset[c],
        exponent, mantissa, clip_min, clip_max, rounding)

        deq = DequantizeScalar(qt, scale[c], offset[c])

        quantized_value[i] = deq

    return quantized_value.reshape(value_size)
    

class ChannelwiseFloatingQuantImpl(Function):
    """Torch Channelwise quantize is designed to quantize a torch Tensor
    with a given configuration. All quantization within PPQ will invoke
    this function to quantize its value. Any modification of this function
    will greatly affects system behaviour.

    This is a torch implementation of quantization itself.
    Notice that if ppq.config.USING_CUDA_KERNAL = True,
        then all quantization will bypass this function by using ffi.CUDA instead.

    Notice this function will always clone your tensor value first.
    This function never quantize your tensor value inplace.
    """
    @ staticmethod
    def forward(ctx, tensor: torch.Tensor, scales: torch.Tensor,
                offsets: torch.Tensor, channel_axis: int,
                exponet_bits: int, mantissa_bits: int, 
                quant_min: float, quant_max: float,
                rounding: RoundingPolicy) -> torch.Tensor:

        scales, offsets = scales.to(tensor.device), offsets.to(tensor.device)
        if not PPQ_CONFIG.USING_CUDA_KERNEL or not tensor.is_cuda:
            # generate a shape that likes [1, 1, -1, 1], the only -1 is at channel axe.
            raise NotImplementedError('This Feature must run with PPQ Cuda Kernel.')
        else:

            from core import CUDA
            quantized = CUDA.FloatingQuantize_C(
                tensor=tensor,
                scales=scales,
                offsets=offsets,
                channel_axis=channel_axis,
                exponent=exponet_bits,
                mantissa=mantissa_bits,
                minimum=quant_min,
                maximum=quant_max,
                rounding=rounding.value)

            # # ****************FP8 python 实现*************************
            # print("*"*40)
            # if exponet_bits <= 0: raise ValueError('Floating Quantization requires exponent > 0')
            # if not tensor.is_contiguous(): tensor = tensor.contiguous()

            # CheckTensor(tensor, torch.float32, "Value(Expect to be FP32)")
            # CheckTensor(scales, torch.float32, "Scale(Expect to be FP32)")
            # CheckTensor(offsets, torch.float32, "Offset(Expect to be FP32)")
            # element_per_channel = 1
            # num_of_channel = tensor.size(channel_axis)
            # for axis in range(tensor.ndimension()-1,channel_axis,-1):
            #     element_per_channel *= tensor.size(axis)
            
            # quantized_python = QuantizeTensor_FC(tensor.numel(), element_per_channel, num_of_channel, 
            # tensor, scales, offsets, exponet_bits, mantissa_bits, quant_min, quant_max, 
            # rounding.value)

            # print(quantized_python == quantized)
            # print((quantized_python == quantized).all())
            # import sys; sys.exit(1)

            # return CUDA_COMPLIER.CUDA_EXTENSION.QuantizeTensor_FC(
            #     tensor, scales, offsets, exponent, mantissa, 
            #     minimum, maximum, channel_axis, rounding)
            return quantized

    @ staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, None, None, None, None, None, None, None, None, None


def PPQFloatingQuantFunction(
    tensor: torch.Tensor, scale: torch.Tensor=None, offset: torch.Tensor=None, 
    channel_axis: int=None, exponent_bits: int=None, mantissa_bits: int=None, 
    quant_min: float=None, quant_max: float=None, rounding: RoundingPolicy=None,
     perchannel: bool=False) -> torch.Tensor:

    if perchannel:
        return ChannelwiseFloatingQuantImpl.apply(
            tensor, scale, offset, channel_axis,
            exponent_bits, mantissa_bits,
            quant_min, quant_max, rounding)
    else:
        return TensorwiseFloatingQuantImpl.apply(
            tensor, scale, offset,
            exponent_bits, mantissa_bits,
            quant_min, quant_max, rounding)

class ENABLE_CUDA_KERNEL:
    """ Any code surrounded by 
    with ENABLE_CUDA_KERNEL():
    will invoke ppq's kernel functions for speed boost.
    
    This is a helper class for invoking highly-effcient custimized cuda
    kernel. PPQ developer team has implemented a series of quantization related
    cuda kernel, They are 5-100x faster than torch kernels, with less gpu
    memory cost.
    """
    def __init__(self) -> None:
        from core.ffi import CUDA_COMPLIER
        CUDA_COMPLIER.complie()
        self._state = False

    def __enter__(self):
        self._state = PPQ_CONFIG.USING_CUDA_KERNEL
        PPQ_CONFIG.USING_CUDA_KERNEL = True

    def __exit__(self, *args):
        PPQ_CONFIG.USING_CUDA_KERNEL = self._state

if __name__ == '__main__':
    import os, sys
    dir = "/your/path/to/fp8_quant_function/csrc/build/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    # import pdb;pdb.set_trace()
    # print(os.path.dirname(os.path.dirname(__file__)))
    tensor = torch.rand([64,128]).cuda()
    # tensor.fill_(-3.8123)
    scale_perchannel = torch.rand([tensor.size(0)]).cuda()
    scale_pertensor = torch.rand([1]).cuda()
    # scale.fill_(0.1)
    offset_perchannel = torch.zeros([tensor.size(0)]).cuda()
    offset_pertensor = torch.zeros([1]).cuda()
    per_channel = True
    with ENABLE_CUDA_KERNEL():
        if per_channel:
            quant_tensor = PPQFloatingQuantFunction(tensor, scale=scale_perchannel, offset=offset_perchannel, 
            channel_axis=0, exponent_bits=4, mantissa_bits=3, 
            quant_min=-448, quant_max=448, rounding=RoundingPolicy.ROUND_HALF_EVEN, 
            perchannel=True)
        else:
            quant_tensor = PPQFloatingQuantFunction(tensor, scale=scale_pertensor, offset=offset_pertensor, 
            channel_axis=-1, exponent_bits=4, mantissa_bits=3, 
            quant_min=-448, quant_max=448, rounding=RoundingPolicy.ROUND_HALF_EVEN, 
            perchannel=False)
    print(tensor)
    print("*"*40)
    print(quant_tensor)
