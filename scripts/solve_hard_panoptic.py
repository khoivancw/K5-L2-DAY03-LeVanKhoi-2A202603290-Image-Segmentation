import json
import zipfile
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

def mask_to_polygons(mask, min_area=40):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Approximate to reduce point count while keeping shape
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        flat = approx.flatten().tolist()
        if len(flat) >= 6:
            polygons.append((flat, area, cv2.boundingRect(approx)))
    return polygons

def main():
    root = Path(__file__).resolve().parents[1]
    task_dir = root / "data" / "tiers" / "hard_panoptic"
    images_dir = task_dir / "images"
    classes_file = task_dir / "classes.json"
    
    classes_data = json.loads(classes_file.read_text(encoding="utf-8"))
    target_classes = classes_data["classes"]
    stuff_classes = set(classes_data["stuff"])
    catid2name = classes_data.get("catid2name", {})
    
    # Invert catid2name
    name2catid = {v: int(k) for k, v in catid2name.items()}
    if not name2catid:
        name2catid = {c: idx for idx, c in enumerate(target_classes, start=1)}
        
    categories = []
    for cls_name in target_classes:
        cid = name2catid[cls_name]
        is_thing = cls_name not in stuff_classes
        categories.append({
            "id": cid,
            "name": cls_name,
            "supercategory": "thing" if is_thing else "stuff",
            "isthing": 1 if is_thing else 0
        })
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading YOLOv8x-seg...")
    yolo_model = YOLO("yolov8x-seg.pt")
    
    print("Loading SegFormer...")
    seg_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    processor = SegformerImageProcessor.from_pretrained(seg_name)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_name).to(device)
    seg_model.eval()
    
    coco_images = []
    coco_annotations = []
    ann_id = 1
    
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    for img_idx, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
            im_rgb = im.convert("RGB")
            
        coco_images.append({
            "id": img_idx,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })
        
        print(f"\nProcessing {img_path.name} ({w}x{h})...")
        
        # 1. Things via YOLOv8x-seg
        thing_names = {"person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light"}
        yolo_results = yolo_model(img_path, conf=0.25, iou=0.5, device=0)
        
        for r in yolo_results:
            if r.masks is None or r.boxes is None:
                continue
            for box, mask_poly in zip(r.boxes, r.masks.xy):
                cls_idx = int(box.cls[0])
                cls_name = yolo_model.names[cls_idx]
                if cls_name not in thing_names or cls_name not in name2catid:
                    continue
                
                cid = name2catid[cls_name]
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
                    "category_id": cid,
                    "segmentation": [flat_poly],
                    "area": float(b[2] * b[3]),
                    "bbox": [round(x_topleft, 2), round(y_topleft, 2), round(b[2], 2), round(b[3], 2)],
                    "iscrowd": 0
                })
                ann_id += 1
                
        # 2. Stuff via SegFormer
        # Cityscapes classes:
        # 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 8: vegetation, 9: terrain, 10: sky
        inputs = processor(images=im_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = seg_model(**inputs)
            upsampled = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
            preds = upsampled.argmax(dim=1)[0].cpu().numpy()
            
        stuff_map = {
            "road": (preds == 0),
            "sidewalk": (preds == 1),
            "building": (preds == 2) | (preds == 3) | (preds == 4),
            "vegetation": (preds == 8) | (preds == 9),
            "sky": (preds == 10)
        }
        
        for s_cls, mask_bin in stuff_map.items():
            if not np.any(mask_bin):
                continue
            cid = name2catid[s_cls]
            polys = mask_to_polygons(mask_bin, min_area=80)
            for flat_poly, area, (bx, by, bw, bh) in polys:
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cid,
                    "segmentation": [flat_poly],
                    "area": float(area),
                    "bbox": [float(bx), float(by), float(bw), float(bh)],
                    "iscrowd": 0
                })
                ann_id += 1
                
        print(f"Total annotations so far: {len(coco_annotations)}")
        
    coco_data = {
        "images": coco_images,
        "categories": categories,
        "annotations": coco_annotations
    }
    
    out_dir = root / "submissions"
    out_dir.mkdir(exist_ok=True)
    out_zip = out_dir / "hard_panoptic.zip"
    
    json_bytes = json.dumps(coco_data, indent=2).encode("utf-8")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/instances_default.json", json_bytes)
        
    print(f"\nGenerated {out_zip} with {len(coco_annotations)} annotations.")

if __name__ == "__main__":
    main()
