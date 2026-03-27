echo "Running turboquant quantization 4 bits"
QUANT_METHOD=turboquant TURBOQUANT_MODE=mse QUANT_BITS=4 python benchmarking.py
echo "Running turboquant quantization 5 bits"
QUANT_METHOD=turboquant TURBOQUANT_MODE=mse QUANT_BITS=5 python benchmarking.py
echo "Running turboquant quantization 6 bits"
QUANT_METHOD=turboquant TURBOQUANT_MODE=mse QUANT_BITS=6 python benchmarking.py
echo "Running turboquant quantization 7 bits"
QUANT_METHOD=turboquant TURBOQUANT_MODE=mse QUANT_BITS=7 python benchmarking.py
echo "Running turboquant quantization 8 bits"
QUANT_METHOD=turboquant TURBOQUANT_MODE=mse QUANT_BITS=8 python benchmarking.py
