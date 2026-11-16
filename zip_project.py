import os
import zipfile
from pathlib import Path

def create_project_zip():
    source_dir = Path(r"C:\Users\heman\Documents\DeepFake_Project\DeepFake Detection Final")
    output_zip = Path(r"C:\Users\heman\Documents\DeepFake_Project\DeepFake_Detection_Final_Submission.zip")
    
    # Exclude massive environments, datasets, and cache
    exclude_dirs = {
        'venv', 
        '.venv',
        'data',
        '.git', 
        '__pycache__', 
        '.pytest_cache',
        '.pytest_tmp',
        '.build',
        'deepfake_detection.egg-info',
        'matplotlib',
        'gradio_temp'
    }
    
    # Exclude unnecessary / redundant files
    exclude_extensions = {'.pyc', '.pyo', '.pyd', '.download', '.tmp'}
    exclude_files = {'last.pt', 'Academic_Files.zip'}  # last.pt is 336MB intermediate epoch; calibrated.pt & best.pt are kept
    
    print(f"Creating zip file at {output_zip}...")
    
    file_count = 0
    total_uncompressed = 0
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue
                if file in exclude_files:
                    continue
                
                file_path = Path(root) / file
                arcname = file_path.relative_to(source_dir.parent)
                
                zipf.write(file_path, arcname)
                file_count += 1
                total_uncompressed += file_path.stat().st_size

    zip_size = output_zip.stat().st_size
    print(f"Zip file created successfully!")
    print(f"Total files packaged: {file_count}")
    print(f"Uncompressed size: {total_uncompressed / (1024*1024):.2f} MB")
    print(f"Compressed ZIP size: {zip_size / (1024*1024):.2f} MB")
    print(f"Location: {output_zip}")

if __name__ == '__main__':
    create_project_zip()

