import os
import shutil
import kagglehub
from config import CHEST_DIR, BRAIN_DIR, DATASET_DIR

def clear_and_recreate_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def download_and_structure_data():
    print("Initializing environment structure...")
    # Base structure
    clear_and_recreate_dir(CHEST_DIR)
    os.makedirs(os.path.join(CHEST_DIR, 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(CHEST_DIR, 'PNEUMONIA'), exist_ok=True)
    os.makedirs(os.path.join(CHEST_DIR, 'TUBERCULOSIS'), exist_ok=True)

    clear_and_recreate_dir(BRAIN_DIR)
    os.makedirs(os.path.join(BRAIN_DIR, 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(BRAIN_DIR, 'TUMOR'), exist_ok=True)

    # 1. Download Chest X-ray (Pneumonia)
    print("Downloading Chest X-ray (Pneumonia) dataset...")
    chest_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    print("Structuring Pneumonia data...")
    # The dataset has chest_xray/train, chest_xray/test, chest_xray/val 
    # and inside are NORMAL and PNEUMONIA folders.
    chest_inner = os.path.join(chest_path, "chest_xray")
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(chest_inner, split)
        if os.path.exists(split_path):
            # Move Normal
            normal_folder = os.path.join(split_path, 'NORMAL')
            if os.path.exists(normal_folder):
                for img in os.listdir(normal_folder):
                    shutil.copy(os.path.join(normal_folder, img), os.path.join(CHEST_DIR, 'NORMAL', img))
            # Move Pneumonia
            pneumonia_folder = os.path.join(split_path, 'PNEUMONIA')
            if os.path.exists(pneumonia_folder):
                for img in os.listdir(pneumonia_folder):
                    shutil.copy(os.path.join(pneumonia_folder, img), os.path.join(CHEST_DIR, 'PNEUMONIA', img))

    # 2. Download Tuberculosis dataset
    print("Downloading Tuberculosis dataset...")
    tb_path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
    
    print("Structuring TB data...")
    # We only take the Tuberculosis images from the TB_Chest_Radiography_Database folder
    tb_folder = os.path.join(tb_path, "TB_Chest_Radiography_Database", "Tuberculosis")
    if os.path.exists(tb_folder):
        for img in os.listdir(tb_folder):
            shutil.copy(os.path.join(tb_folder, img), os.path.join(CHEST_DIR, 'TUBERCULOSIS', img))

    # 3. Download Brain MRI
    print("Downloading Brain MRI dataset...")
    brain_path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
    
    print("Structuring Brain data...")
    # Extract 'yes' -> TUMOR, 'no' -> NORMAL
    yes_path = os.path.join(brain_path, "brain_tumor_dataset", "yes")
    no_path = os.path.join(brain_path, "brain_tumor_dataset", "no")
    
    if os.path.exists(yes_path):
        for img in os.listdir(yes_path):
            shutil.copy(os.path.join(yes_path, img), os.path.join(BRAIN_DIR, 'TUMOR', img))
            
    if os.path.exists(no_path):
        for img in os.listdir(no_path):
            shutil.copy(os.path.join(no_path, img), os.path.join(BRAIN_DIR, 'NORMAL', img))

    print("Data consolidation complete. You are ready to train.")

if __name__ == "__main__":
    download_and_structure_data()
