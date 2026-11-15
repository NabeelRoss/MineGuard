import sys
import importlib
import cv2
import torch
import numpy as np

print("=" * 60)
print("          MINEGUARD AI - ENVIRONMENT TEST")
print("=" * 60)

results = {}

# 1. Python
print("\n[1] PYTHON")
print("Version:", sys.version)
results["Python"] = True

# 2. PyTorch
print("\n[2] PYTORCH")
print("Version:", torch.__version__)
results["PyTorch"] = True

# 3. CUDA
print("\n[3] CUDA / NVIDIA GPU")
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
results["CUDA"] = cuda_available

if cuda_available:
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

    # GPU computation test
    try:
        a = torch.rand(1000, 1000, device="cuda")
        b = torch.rand(1000, 1000, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("GPU Computation: SUCCESS")
        results["GPU Computation"] = True
    except Exception as e:
        print("GPU Computation: FAILED -", e)
        results["GPU Computation"] = False
else:
    print("WARNING: Running without GPU acceleration.")

# 4. OpenCV
print("\n[4] OPENCV")
print("Version:", cv2.__version__)
results["OpenCV"] = True

# 5. NumPy
print("\n[5] NUMPY")
print("Version:", np.__version__)
results["NumPy"] = True

# 6. Additional libraries
print("\n[6] ADDITIONAL LIBRARIES")

libraries = [
    "torchvision",
    "ultralytics",
    "matplotlib",
    "pandas",
    "sklearn",
    "yaml",
    "tqdm"
]

for library in libraries:
    try:
        module = importlib.import_module(library)
        version = getattr(module, "__version__", "Installed")
        print(f"{library}: OK ({version})")
        results[library] = True
    except Exception as e:
        print(f"{library}: FAILED ({e})")
        results[library] = False

# 7. YOLO11 model test
print("\n[7] YOLO11 MODEL TEST")

try:
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    print("YOLO11n: Loaded successfully")

    # Dummy inference test
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)

    prediction = model.predict(
        source=test_image,
        device=0 if cuda_available else "cpu",
        verbose=False
    )

    print("YOLO Inference: SUCCESS")
    results["YOLO Model"] = True

except Exception as e:
    print("YOLO Test FAILED:", e)
    results["YOLO Model"] = False

# 8. Webcam test
print("\n[8] WEBCAM TEST")

try:
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print("Webcam: Connected")
            print("Resolution:", frame.shape[1], "x", frame.shape[0])
            results["Webcam"] = True
        else:
            print("Webcam opened but frame capture failed")
            results["Webcam"] = False
    else:
        print("Webcam: Not available")
        results["Webcam"] = False

    cap.release()

except Exception as e:
    print("Webcam Error:", e)
    results["Webcam"] = False

# 9. Final report
print("\n" + "=" * 60)
print("                 FINAL TEST REPORT")
print("=" * 60)

for component, status in results.items():
    print(f"{component:<25} : {'PASS' if status else 'FAIL'}")

passed = sum(results.values())
total = len(results)

print("\nTotal Tests:", total)
print("Passed:", passed)
print("Failed:", total - passed)

if all(results.values()):
    print("\nMINEGUARD AI ENVIRONMENT READY!")
else:
    print("\nSOME COMPONENTS NEED ATTENTION.")

print("=" * 60)
