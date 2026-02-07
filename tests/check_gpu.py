import tensorflow as tf
import lightgbm as lgb
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    print("========================================")
    print("🔍 Checking Hardware Acceleration Check")
    print("========================================")
    
    # ---------------------------------------------------------
    # 1. TensorFlow Check (Critical for LSTM/Deep Learning)
    # ---------------------------------------------------------
    print("\n[1] TensorFlow")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ SUCCESS: Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                gpu_name = details.get('device_name', 'Unknown')
                print(f"   🚀 Device: {gpu.name} -> {gpu_name}")
        else:
            print("⚠️  WARNING: No GPU found for TensorFlow (Standard CPU will be used)")
    except Exception as e:
        print(f"❌ ERROR: TensorFlow check failed: {e}")

    # ---------------------------------------------------------
    # 2. LightGBM Check
    # ---------------------------------------------------------
    print("\n[2] LightGBM")
    
    # Prepare dummy data
    data = np.random.rand(50, 10)
    label = np.random.rand(50)
    train_data = lgb.Dataset(data, label=label)

    # 2.1 Test GPU (Expect failure if OpenCL is missing)
    print("   Testing GPU mode...", end=" ")
    try:
        params_gpu = {
            'device': 'cuda',  # Standard parameter name
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbose': -1,
            'force_col_wise': True
        }
        lgb.train(params_gpu, train_data, num_boost_round=1)
        print("✅ SUCCESS! (LightGBM is using GPU)")
    except Exception as e:
        print("❌ FAILED")
        print(f"      Reason: {str(e).strip().split('\n')[0]}") # Show first line of error
        print("      (Standard behavior for pip-installed LightGBM on WSL without OpenCL)")

    # 2.2 Test CPU (Fallback - Must pass)
    print("   Testing CPU mode...", end=" ")
    try:
        params_cpu = {
            'device': 'cpu',
            'verbose': -1
        }
        lgb.train(params_cpu, train_data, num_boost_round=1)
        print("✅ SUCCESS! (LightGBM works on CPU)")
    except Exception as e:
        print("❌ CRITICAL FAILURE: LightGBM cannot run at all.")
        print(f"      Error: {e}")

    print("\n========================================")

if __name__ == "__main__":
    check_gpu()