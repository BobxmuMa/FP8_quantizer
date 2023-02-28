# FP8_quantizer

通过如下命令安装环境:

    git clone https://github.com/BobxmuMa/FP8_quantizer.git
    cd FP8_quantizer
    pip install -r fp8_quant_function/requirements.txt
    
运行FP8量化器：

`python3 fp8_quant_function/floating.py`

里面同时包含了python版本的FP8量化器实现，但被注释掉了。其量化结果和c++版本一致。但c++版本通过CUDA编程，可调用CUDA内核并行计算，FP8量化效率很高。而python版本量化器由于循环的存在计算效率低。


# Reference

https://github.com/openppl-public/ppq
