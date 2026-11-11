import json
import zipfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

def process_checkpoint(model, cp_name, root):
    cp_dir = root / "data" / "checkpoints" / cp_name
    images_dir = cp_dir / "images"
    classes_file = cp_dir / "classes.json"
    
    classes_data = json.loads(classes_file.read_text(encoding="utf-8"))
    target_classes = classes_data["classes"]
    
    categories = []
    class_to_cat_id = {}
    for idx, cls_name in enumerate(target_classes, start=1):
        categories.append({"id": idx, "name": cls_name, "supercategory": "none"})
        class_to_cat_id[cls_name] = idx
        
    coco_images = []
    coco_annotations = []
    ann_id = 1
    
    image_files = sorted(list(images_dir.glob("*.jpg")))
    for img_idx, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
            
        coco_images.append({
            "id": img_idx,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })
        
        print(f"[{cp_name}] Inferencing {img_path.name} ({w}x{h})...")
        results = model(img_path, conf=0.25, iou=0.5, device=0)
        
        for r in results:
            if r.masks is None or r.boxes is None:
                continue
            for box, mask_poly in zip(r.boxes, r.masks.xy):
                cls_idx = int(box.cls[0])
                cls_name = model.names[cls_idx]
                if cls_name not in class_to_cat_id:
                    continue
                
                cat_id = class_to_cat_id[cls_name]
                
                poly = mask_poly.tolist()
                if len(poly) < 3:
                    continue
                flat_poly = [coord for pt in poly for coord in pt]
                if len(flat_poly) < 6:
                    continue
                
                b = box.xywh[0].tolist()
                x_topleft = b[0] - b[2] / 2
                y_topleft = b[1] - b[3] / 2
                
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_id,
                    "segmentation": [flat_poly],
                    "area": float(b[2] * b[3]),
                    "bbox": [round(x_topleft, 2), round(y_topleft, 2), round(b[2], 2), round(b[3], 2)],
                    "iscrowd": 0
                })
                ann_id += 1
                
    coco_data = {
        "images": coco_images,
        "categories": categories,
        "annotations": coco_annotations
    }
    
    out_dir = root / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_zip = out_dir / f"{cp_name}.zip"
    
    json_bytes = json.dumps(coco_data, indent=2).encode("utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/instances_default.json", json_bytes)
        
    print(f"-> Generated {out_zip} with {len(coco_annotations)} instances.")

def main():
    root = Path(__file__).resolve().parents[1]
    print("Loading YOLOv8x-seg model...")
    model = YOLO("yolov8x-seg.pt")
    
    for cp in ["cp1_holes", "cp2_slice", "cp5_occlusion"]:
        process_checkpoint(model, cp, root)

if __name__ == "__main__":
    main()
