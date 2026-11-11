import json
import zipfile
import io
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

def main():
    root = Path(__file__).resolve().parents[1]
    checkpoints_dir = root / "data" / "checkpoints"
    out_dir = root / "submissions"
    out_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    print("Loading SegFormer model...")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    
    # Cityscapes label mapping:
    # 0: road, 1: sidewalk, 2: building, 3: wall, 4: fence, 5: pole, 6: traffic light,
    # 7: traffic sign, 8: vegetation, 9: terrain, 10: sky, 11: person, 12: rider,
    # 13: car, 14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle
    
    tasks = ["cp3_thin", "cp4_curb", "cp6_coverage"]
    
    for task_name in tasks:
        task_dir = checkpoints_dir / task_name
        classes_data = json.loads((task_dir / "classes.json").read_text(encoding="utf-8"))
        target_classes = classes_data["classes"]
        colors = classes_data["colors"]
        
        # Build labelmap.txt content
        labelmap_lines = ["# label:color_rgb:parts:actions", "background:0,0,0::"]
        for cls_name in target_classes:
            c = colors[cls_name]
            labelmap_lines.append(f"{cls_name}:{c[0]},{c[1]},{c[2]}::")
        labelmap_text = "\n".join(labelmap_lines) + "\n"
        
        image_files = sorted(list((task_dir / "images").glob("*.jpg")))
        
        out_zip = out_dir / f"{task_name}.zip"
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("labelmap.txt", labelmap_text.encode("utf-8"))
            
            for img_path in image_files:
                im = Image.open(img_path).convert("RGB")
                w, h = im.size
                
                inputs = processor(images=im, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    upsampled_logits = F.interpolate(
                        logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False
                    )
                    preds = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
                
                # Create RGB mask initialized to black (0, 0, 0)
                mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                
                if task_name == "cp3_thin":
                    # ['pole', 'traffic sign', 'sky', 'road']
                    # pole=5, traffic sign=7, sky=10, road=0
                    mask_rgb[preds == 5] = colors["pole"]
                    mask_rgb[preds == 7] = colors["traffic sign"]
                    mask_rgb[preds == 10] = colors["sky"]
                    mask_rgb[preds == 0] = colors["road"]
                    
                elif task_name == "cp4_curb":
                    # ['road', 'sidewalk']
                    mask_rgb[preds == 0] = colors["road"]
                    mask_rgb[preds == 1] = colors["sidewalk"]
                    
                elif task_name == "cp6_coverage":
                    # ['road', 'sidewalk', 'building', 'vegetation', 'sky', 'car', 'person']
                    # Ensure full coverage (no unlabeled gaps)
                    # Direct mappings:
                    mask_rgb[preds == 0] = colors["road"]
                    mask_rgb[preds == 1] = colors["sidewalk"]
                    mask_rgb[preds == 2] = colors["building"]
                    mask_rgb[preds == 8] = colors["vegetation"]
                    mask_rgb[preds == 10] = colors["sky"]
                    mask_rgb[preds == 13] = colors["car"]
                    mask_rgb[preds == 11] = colors["person"]
                    
                    # Semantically map secondary cityscapes classes to the 7 target classes:
                    mask_rgb[preds == 3] = colors["building"]   # wall -> building
                    mask_rgb[preds == 4] = colors["building"]   # fence -> building
                    mask_rgb[preds == 5] = colors["building"]   # pole -> building
                    mask_rgb[preds == 6] = colors["building"]   # traffic light -> building
                    mask_rgb[preds == 7] = colors["building"]   # traffic sign -> building
                    mask_rgb[preds == 9] = colors["vegetation"] # terrain -> vegetation
                    mask_rgb[preds == 12] = colors["person"]    # rider -> person
                    mask_rgb[preds == 14] = colors["car"]       # truck -> car
                    mask_rgb[preds == 15] = colors["car"]       # bus -> car
                    mask_rgb[preds == 16] = colors["car"]       # train -> car
                    mask_rgb[preds == 17] = colors["car"]       # motorcycle -> car
                    mask_rgb[preds == 18] = colors["car"]       # bicycle -> car
                    
                    # Any remaining black pixel (if any) -> road/building default to achieve 100% coverage
                    black_pixels = np.all(mask_rgb == [0, 0, 0], axis=-1)
                    if np.any(black_pixels):
                        mask_rgb[black_pixels] = colors["road"]
                        
                mask_pil = Image.fromarray(mask_rgb)
                buf = io.BytesIO()
                mask_pil.save(buf, format="PNG")
                z.writestr(f"SegmentationClass/{img_path.stem}.png", buf.getvalue())
                print(f"[{task_name}] Processed {img_path.name}")
                
        print(f"-> Generated {out_zip}")

if __name__ == "__main__":
    main()
