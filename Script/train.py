import os
from datetime import datetime
from ultralytics import YOLO
#CONFIG
CFG = {
    #Dataset - classification
    "DATA_DIR": r"D:\CODE\AI\TomatoLeafDisease\Dataset",
    #Mô hình
    "MODEL": "yolov8m-cls.pt",
    #Tham số train
    "IMGSZ": 224,
    "EPOCHS": 50,
    "BATCH": 64,
    "WORKERS": 8,
    "PATIENCE": 0,         
    "LABEL_SMOOTHING": 0.05,
    "SEED": 42,
    "DEVICE": None,
    "PROJECT": "runs/classify",
    "NAME": "tomato_v1",
    "EXPORT_ONNX": False,
}
def _assert_dataset_structure(data_dir: str):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    valid_dir = os.path.join(data_dir, "valid")
    if not os.path.isdir(train_dir):
        raise SystemExit(
            f"[!] Không thấy thư mục train: {train_dir}\n"
            "    Yêu cầu cấu trúc: <DATA_DIR>/train/<class_folders>"
        )
    if not (os.path.isdir(val_dir) or os.path.isdir(valid_dir)):
        raise SystemExit(
            f"[!] Không thấy thư mục val/valid trong: {data_dir}\n"
            "    Tạo thư mục 'val' (hoặc 'valid') và đặt ảnh theo lớp."
        )

def main():
    data_dir = CFG["DATA_DIR"]
    _assert_dataset_structure(data_dir)

    model_name = CFG["MODEL"]
    imgsz = CFG["IMGSZ"]
    epochs = CFG["EPOCHS"]
    batch = CFG["BATCH"]
    workers = CFG["WORKERS"]
    patience = CFG["PATIENCE"]
    label_smoothing = CFG["LABEL_SMOOTHING"]
    seed = CFG["SEED"]
    device = CFG["DEVICE"]
    project = CFG["PROJECT"]
    name = CFG["NAME"] or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    export_onnx = CFG["EXPORT_ONNX"]

    print("[i] Dataset :", data_dir)
    print("[i] Model   :", model_name)
    print("[i] Save to :", os.path.join(project, name))

    model = YOLO(model_name)

    # Train
    results = model.train(
        data=data_dir,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        patience=patience,
        seed=seed,
        label_smoothing=label_smoothing,
        project=project,
        name=name,
        device=device,
        verbose=True,
    )
    # Val nhanh để in top-1
    metrics = model.val(data=data_dir, imgsz=imgsz, device=device)
    try:
        top1 = metrics.results_dict.get("metrics/accuracy_top1", None)
        if top1 is not None:
            print(f"[✓] Val accuracy@top1: {top1:.4f}")
    except Exception:
        print("[i] Val xong. Xem chi tiết trong thư mục runs.")
    # Xuất ONNX
    if export_onnx:
        print("[i] Export ONNX...")
        model.export(format="onnx", opset=13, dynamic=False)
    # Vị trí lưu
    try:
        save_dir = model.trainer.save_dir
    except Exception:
        save_dir = os.path.join(project, name)
    print(f"[✓] DONE. Kết quả ở: {save_dir}")

if __name__ == "__main__":
    main()
