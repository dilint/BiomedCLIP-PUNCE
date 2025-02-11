import torch.nn as nn

class LinearAdapter(nn.Module):
    def __init__(self, input_dim):
        super(LinearAdapter, self).__init__()
        self.hide_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        x = self.hide_layer(x)
        return x


if __name__ == '__main__':
    import torch
    
    model = LinearAdapter(512)
    ckp_path = '/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/output-model/simclr-infonce/biomedclip_simclr_infonce_filtergc_50_224_4*256/biomedclip_simclr_infonce_filtergc_50_224_4*256_epoch200.pt'
    ckp = torch.load(ckp_path)
    info = model.load_state_dict(ckp['adapter'])
    print(info)