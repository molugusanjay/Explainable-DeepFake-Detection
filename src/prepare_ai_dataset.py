import os
import csv
import hashlib
import random
from pathlib import Path

def generate_ai_manifest(output_csv="data/prepared/manifest.csv"):
    base_dir = Path("data/ai_dataset/140k Real and Fake Faces/real_vs_fake/real-vs-fake/train")
    if not base_dir.exists():
        print(f"Error: Could not find the 140k dataset folder at {base_dir}")
        return

    rows = []
    
    # We expect two subdirectories: 'real' and 'fake'
    for label_str, class_name in [('0', 'real'), ('1', 'fake')]:
        folder = base_dir / class_name
        if not folder.exists():
            print(f"Warning: Missing folder {folder}. Skipping.")
            continue
            
        images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        
        # DOWN-SAMPLE FOR CPU TRAINING
        # The user is running on CPU. Training 140,000 images on CPU takes months.
        # We sample exactly 1000 images per class (2000 total) for a few hours of training time.
        random.shuffle(images)
        images = images[:1000]
        
        for i, img_path in enumerate(images):
            # Assign splits: 80% train, 20% val
            rand = random.random()
            if rand < 0.8:
                split = 'train'
            else:
                split = 'val'
                
            # Define destination path
            dest_folder = Path(output_csv).parent / 'images' / class_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            dest_path = dest_folder / img_path.name
            
            # Copy image to prepared directory so data.py accepts it
            import shutil
            shutil.copy2(img_path, dest_path)
                
            sha256 = hashlib.sha256(dest_path.read_bytes()).hexdigest()
            rel_path = os.path.relpath(dest_path, start=Path(output_csv).parent)
            rel_path = rel_path.replace('\\', '/')
            
            rows.append({
                'path': rel_path,
                'label': label_str,
                'split': split,
                'group': f'group_{class_name}_{i}',
                'source_ids': f'src_{class_name}_{i}',
                'sha256': sha256,
                'method': 'ai_generated' if class_name == 'fake' else 'real_photo'
            })

    if not rows:
        print("No images found!")
        return
        
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Successfully created manifest with {len(rows)} images at {output_csv}!")
    print("Because you are on CPU, the dataset was scaled down to 2,000 images to prevent freezing.")
    print("You can now run: python src/train.py")

if __name__ == '__main__':
    generate_ai_manifest()
