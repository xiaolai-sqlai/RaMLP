import torch, math
from thop import profile, clever_format

from models.ramlp import RaMLP



if __name__=="__main__":
    input = torch.randn(1, 3, 224, 224)
    model = RaMLP(depths=[3, 3, 12, 3], dims=[64, 128, 256, 512], mlp_ratio=[4, 4, 3, 3], expansion_ratio=[3, 3, 2, 2], kernel_size=[8, 4, 2, 1], head_dims=[1, 4, 16, 64], drop_path_rate=0.2)
    model = RaMLP(depths=[3, 8, 26, 3], dims=[64, 128, 256, 512], mlp_ratio=[4, 4, 3, 3], expansion_ratio=[3, 3, 2, 2], kernel_size=[8, 4, 2, 1], head_dims=[1, 4, 16, 64], drop_path_rate=0.3)
    model = RaMLP(depths=[3, 8, 26, 3], dims=[80, 160, 320, 640], mlp_ratio=[4, 4, 3, 3], expansion_ratio=[3, 3, 2, 2], kernel_size=[8, 4, 2, 1], head_dims=[1, 4, 16, 64], drop_path_rate=0.4)

    model.eval()
    print(model)

    macs, params = profile(model, inputs=(input, ), custom_ops={})
    macs, params = clever_format([macs, params], "%.3f")

    params = sum(p.numel() for p in model.parameters()) / 1e6

    print('Flops:  ', macs)
    print('Params: ', params)

