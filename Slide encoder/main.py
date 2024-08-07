import torch
from torch.utils.data import Dataset
import os, glob
from gigapath_slide_encoder import create_model
import torch.nn.functional as F
from tqdm import tqdm

class GcDataset(Dataset):
    def __init__(self, wsi_names, img_root, feature_root):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(GcDataset, self).__init__()
        self.wsi_names = wsi_names
        self.img_root = img_root
        self.feature_root = feature_root
        
    def __len__(self):
        return len(self.wsi_names)
        
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        feature_root = os.path.join(self.feature_root,"pt")
        wsi_path = os.path.join(feature_root, self.wsi_names[idx]+'.pt')
        features = torch.load(wsi_path)
        if os.path.exists(os.path.join(self.img_root, 'NILM', self.wsi_names[idx])):
            wsi_img_path = os.path.join(self.img_root, 'NILM', self.wsi_names[idx])
        else:
            wsi_img_path = os.path.join(self.img_root, 'POS', self.wsi_names[idx])
        patch_files = glob.glob(os.path.join(wsi_img_path, '*.jpg'))
        patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
        int(os.path.basename(x).split(".")[0].split("_")[1])))
        coordinates = []
        for f in patch_files:
            x = int(os.path.basename(f).split(".")[0].split("_")[0])
            y = int(os.path.basename(f).split(".")[0].split("_")[1])
            coordinates.append(torch.tensor([x,y]))
        coordinates = torch.stack(coordinates, dim=0)
        
        return features, coordinates

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='MIL Training Script')
    
    label_path = '../datatools/gc/labels/all_label.csv'
    label_path = '/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc/labels/all_label.csv'
    img_root = '/home1/wsi/gc-224'
    feature_root = '/home1/wsi/gc-all-features/frozen/gigapath1'
    output_root = '/home1/wsi/gc-all-features/frozen/gigapath-longnet'
    if not os.path.exists(output_root):
            os.makedirs(output_root)
    wsi_names = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            wsi_names.append(line.split(',')[0])
    dataset = GcDataset(wsi_names, img_root, feature_root)
    print(dataset[0])

    slide_encoder = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, download=True)
    slide_encoder.eval()
    slide_encoder.to('cuda')
    for i, wsi_name in tqdm(enumerate(wsi_names)):
        tile_embeds, coords = dataset[i]
        if len(tile_embeds.shape) == 2:
            tile_embeds = tile_embeds.unsqueeze(0)
            coords = coords.unsqueeze(0)
        # run inference
        with torch.cuda.amp.autocast(dtype=torch.float16):
            slide_embeds = slide_encoder(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
        outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
        outputs_feat = slide_embeds[11].detach().cpu().squeeze()
        output_dir = os.path.join(output_root, f'{wsi_name}.pt')
        torch.save(outputs_feat, output_dir)
    