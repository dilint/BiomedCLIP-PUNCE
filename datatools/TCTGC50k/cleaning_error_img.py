import os, glob
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

root_path = '/data/wsi/TCTGC50k/TCTGC50k-volume-deprecated45'
root_path = '/data/wsi/TCTGC50k/TCTGC50k-volume4'
wsi_names = os.listdir(root_path)

def load_img(img_path):
    # img = Image.open(img_path)
    try:
        with Image.open(img_path) as img:
            # img.load()
            # img.resize((224, 224))
            transforms.Resize((112,112))(img)
    except Exception as e:
        print(f"[ERROR] {os.path.basename(os.path.dirname(img_path))} {e}")

def load_wsi(wsi_path):
    try:
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + glob.glob(os.path.join(wsi_path, '*.png'))
        # sort to ensure reproducibility
        patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), int(os.path.basename(x).split(".")[0].split("_")[1])))
    except Exception as e:
        print(f'[ERROR] {os.path.basename(wsi_path)} {wsi_path}')
        print(e)
        return
    
    with ThreadPoolExecutor(max_workers=40) as executor:
        executor.map(load_img, [os.path.join(wsi_path, patch_file) for patch_file in patch_files])

    
for wsi_name in tqdm(wsi_names):
    load_wsi(os.path.join(root_path, wsi_name))
    
