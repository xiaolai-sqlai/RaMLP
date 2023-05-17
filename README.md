# RaMLP: Vision MLP via Region-aware Mixing

You could reproduce the model by the code.
```
nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model tiny --drop_path 0.2 --epochs 300 --batch_size 128 --lr 4.0e-3 --update_freq 4 --model_ema false --model_ema_eval false --use_amp true --data_path /INPUT/dataset/imagenet --output_dir ./checkpoint &

nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model small --drop_path 0.3 --epochs 300 --batch_size 128 --lr 4.0e-3 --update_freq 4 --model_ema false --model_ema_eval false --use_amp true --data_path /INPUT/dataset/imagenet --output_dir ./checkpoint &

nohup python -u -m torch.distributed.run --nproc_per_node=8 main.py --model base --drop_path 0.4 --epochs 300 --batch_size 128 --lr 4.0e-3 --update_freq 4 --model_ema false --model_ema_eval false --use_amp true --data_path /INPUT/dataset/imagenet --output_dir ./checkpoint &
```

## Results and Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| RaMLP-T | 224x224 | 82.9 | 25M | 4.2G | [model](https://pan.baidu.com/s/1Ylo1DiJTHGtDYgOVmojCJQ?pwd=c88f) |
| RaMLP-S | 224x224 | 83.8 | 38M | 7.8G | model |
| RaMLP-B | 224x224 | 84.1 | 58M | 12.0G | model |

## Citation
If you find this repository helpful, please consider citing:
```
@Article{liu2022convnet,
  author  = {Shenqi Lai, Xi Du and Jia Guo and Kaipeng Zhang},
  title   = {RaMLP: Vision MLP via Region-aware Mixing},
  journal = {IJCAI},
  year    = {2023},
}
```