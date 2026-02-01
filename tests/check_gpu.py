
import tensorflow as tf
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    print("🔍 Checking GPU Availability...")
    
    # 1. TensorFlow Check
    print("\n[TensorFlow]")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("❌ No GPU found for TensorFlow (will use CPU)")

    # 2. LightGBM Check (indirect)
    print("\n[LightGBM]")
    try:
        # Simple training attempt with gpu
        import numpy as np
        data = np.random.rand(100, 10)
        label = np.random.rand(100)
        train_data = lgb.Dataset(data, label=label)
        params = {'device_type': 'gpu', 'verbose': -1}
        lgb.train(params, train_data, num_boost_round=1)
        print("✅ LightGBM GPU training test passed!")
    except Exception as e:
        print(f"❌ LightGBM GPU failed: {e}")
        print("   (You may need to reinstall lightgbm with 'pip install lightgbm --install-option=--gpu')")

if __name__ == "__main__":
    check_gpu()
