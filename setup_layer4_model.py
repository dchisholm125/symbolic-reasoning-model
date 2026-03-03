import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import time

try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
except ImportError:
    print("Please install requirements first:")
    print("pip install transformers optimum[onnxruntime] onnxruntime scipy numpy")
    exit(1)

def export_model(save_directory="./layer4_onnx_model"):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    
    print(f"Downloading and converting '{model_id}' to ONNX format...")
    print(f"Destination: {save_directory}")
    print("This may take a minute or two depending on your connection...\n")
    
    start_time = time.time()
    
    # AutoTokenizer downloads the fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # ORTModelForFeatureExtraction downloads the PyTorch model and converts it to ONNX (export=True)
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    
    # Save the ONNX model, configs, and tokenizer to our local directory
    os.makedirs(save_directory, exist_ok=True)
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    elapsed = time.time() - start_time
    print(f"\nModel successfully converted and saved in {elapsed:.1f} seconds!")
    print("\nFiles generated:")
    for f in os.listdir(save_directory):
        size_mb = os.path.getsize(os.path.join(save_directory, f)) / (1024 * 1024)
        print(f" - {f} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert MiniLM to ONNX for SRM Layer 4.")
    parser.add_argument("--output", type=str, default="./layer4_onnx_model", help="Directory to save the ONNX model.")
    args = parser.parse_args()
    
    export_model(args.output)
