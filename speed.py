# =============================================================================
#           STANDARDIZED MODEL PERFORMANCE BENCHMARK SCRIPT
#                         (Adapted for DFMG)
# =============================================================================

import os
import time
import sys
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Dependency Imports with Graceful Fallbacks ---
try:
    from ptflops import get_model_complexity_info

    ptflops_available = True
except ImportError:
    ptflops_available = False
    print("Warning: ptflops not found. FLOPs calculation with ptflops will be skipped.")

try:
    from fvcore.nn import parameter_count, parameter_count_table

    fvcore_available = True
except ImportError:
    fvcore_available = False
    print("Warning: fvcore not found. Parameter count with fvcore will be skipped.")

try:
    from torchprofile import profile_macs

    torchprofile_available = True
except ImportError:
    torchprofile_available = False
    print("Warning: torchprofile not found. FLOPs calculation with torchprofile will be skipped.")

try:
    from thop import profile as thop_profile

    thop_available = True
except ImportError:
    thop_available = False
    print("Warning: thop not found. FLOPs calculation with thop will be skipped.")

# --- Suppress Warnings and Set Environment ---
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

print(f"Running on device: {device}")

# =============================================================================
#                       1. MODEL CONFIGURATION
# =============================================================================
# Standard benchmark input size
BENCHMARK_SIZE = (384, 384)
# The native input size from the training script (opt.trainsize).
MODEL_NATIVE_SIZE = (352, 352)
# Path to pretrained weights, if any. Set to None for random initialization.
PRETRAINED_PATH = None

# =============================================================================
#                       2. MODEL INITIALIZATION
# =============================================================================
model = None
try:
    # Assuming the model class 'DFMG' is in 'models/DFMG.py'
    from models.DFMG import DFMG

    print(f"Initializing model: DFMG")
    model = DFMG().to(device)

    if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH))
        print(f"Loaded pretrained weights from: {PRETRAINED_PATH}")
    else:
        print("Initializing model with random weights (no pretrained path provided).")
    model.eval()
    print("Model initialized successfully.")
except Exception as e:
    print(f"Fatal: Model initialization failed. Ensure 'models/DFMG.py' is in the correct path.")
    print(f"Error: {e}")
    sys.exit(1)


# =============================================================================
#               3. BENCHMARKING WRAPPER AND FORWARD CALL
# =============================================================================
class BenchmarkWrapper(nn.Module):
    """
    Wraps the DFMG model to standardize the input and output for benchmarking.
    It handles resizing and the specific depth map replication logic.
    """

    def __init__(self, model, target_size):
        super().__init__()
        self.model = model
        self.target_size = target_size

    def forward(self, x):
        # 1. Resize standard RGB input to the model's native size
        if x.shape[2:] != self.target_size:
            x_resized = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        else:
            x_resized = x

        # 2. Create a dummy single-channel depth map
        dummy_depth_1ch = torch.randn(x_resized.shape[0], 1, *self.target_size, device=x.device)

        # 3. Replicate depth map to 3 channels, as done in the training script
        dummy_depth_3ch = dummy_depth_1ch.repeat(1, 3, 1, 1)

        # 4. Call the model with the two separate inputs (image, depth)
        p, _pg, _h = self.model(x_resized, dummy_depth_3ch)

        # 5. Return the primary prediction 'p' for benchmarking
        return p


# Instantiate the wrapper
wrapper = BenchmarkWrapper(model, MODEL_NATIVE_SIZE).to(device)
wrapper.eval()

# =============================================================================
#           4. & 5. PERFORMANCE METRICS & SUMMARY (CORRECTED VERSION)
# =============================================================================
dummy_input = torch.randn(1, 3, *BENCHMARK_SIZE, device=device)

# --- Parameters (Corrected to count ALL parameters with fallbacks) ---
print("\n============== Parameters ==============")
# fvcore's parameter_count already includes all parameters (trainable and non-trainable)
if fvcore_available:
    try:
        params_m = parameter_count(model)[""] / 1e6
        print(parameter_count_table(model))
        print(f"Total Params (all): {params_m:.2f} M")
    except Exception as e:
        print(f"Parameter count calculation with fvcore failed: {e}")
        # Fallback to the corrected PyTorch method if fvcore fails
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Total Params (all, PyTorch fallback): {params_m:.2f} M")
else:
    # CORRECTED: This counts ALL parameters, not just trainable ones.
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Params (all, PyTorch method): {params_m:.2f} M")


# --- FLOPs --- (No changes needed)
print("\n============== FLOPs ==============")
flops_torchprofile_g, flops_thop_g, flops_ptflops_g = 0, 0, 0

if torchprofile_available:
    try:
        macs = profile_macs(wrapper, (dummy_input,))
        flops_torchprofile_g = macs * 2 / 1e9
        print(f"FLOPs (torchprofile): {flops_torchprofile_g:.2f} GFLOPs")
    except Exception as e:
        print(f"FLOPs calculation (torchprofile) failed: {e}")

if thop_available and flops_torchprofile_g == 0:
    try:
        macs, _ = thop_profile(wrapper, inputs=(dummy_input,), verbose=False)
        flops_thop_g = macs * 2 / 1e9
        print(f"FLOPs (thop): {flops_thop_g:.2f} GFLOPs")
    except Exception as e:
        print(f"FLOPs calculation (thop) failed: {e}")

if ptflops_available and flops_torchprofile_g == 0 and flops_thop_g == 0:
    try:
        macs, _ = get_model_complexity_info(wrapper, (3, *BENCHMARK_SIZE), as_strings=False, print_per_layer_stat=False,
                                            verbose=False)
        flops_ptflops_g = macs * 2 / 1e9
        print(f"FLOPs (ptflops): {flops_ptflops_g:.2f} GFLOPs")
    except Exception as e:
        print(f"FLOPs calculation (ptflops) failed: {e}")

# --- Inference FPS --- (No changes needed)
print("\n============== Inference FPS ==============")
fps = 0.0
try:
    with torch.no_grad():
        for _ in range(20):
            _ = wrapper(dummy_input)
        torch.cuda.synchronize()

        start_time = time.time()
        iterations = 100
        for _ in range(iterations):
            _ = wrapper(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()

    elapsed_time = end_time - start_time
    fps = iterations / elapsed_time if elapsed_time > 0 else float('inf')
    print(f"Average FPS: {fps:.2f}")
except Exception as e:
    print(f"Inference FPS calculation failed: {e}")

# =============================================================================
#                              5. FINAL SUMMARY
# =============================================================================
print("\n\n" + "=" * 30)
print("       PERFORMANCE SUMMARY")
print("=" * 30)
print(f"Model Name:          DFMG")
print(f"Benchmark Input Size: {BENCHMARK_SIZE[0]}x{BENCHMARK_SIZE[1]}")
print(f"Model Native Size:    {MODEL_NATIVE_SIZE[0]}x{MODEL_NATIVE_SIZE[1]}")
if 'params_m' in locals() and params_m > 0:
    print(f"Total Params (all):   {params_m:.2f} M")
else:
    print("Total Params (all):   Not Available")

if flops_torchprofile_g > 0:
    print(f"Total FLOPs:          {flops_torchprofile_g:.2f} GFLOPs (torchprofile)")
elif flops_thop_g > 0:
    print(f"Total FLOPs:          {flops_thop_g:.2f} GFLOPs (thop)")
elif flops_ptflops_g > 0:
    print(f"Total FLOPs:          {flops_ptflops_g:.2f} GFLOPs (ptflops)")
else:
    print("Total FLOPs:          Not Available")
print(f"Average FPS:          {fps:.2f}")
print("=" * 30 + "\n")