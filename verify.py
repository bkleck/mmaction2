import torch
from mmaction.apis import init_recognizer, inference_recognizer

print(f'Torch version: {torch. __version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU device: {torch.cuda.get_device_name(0)}')
config_file = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device)
# inference the demo video
result = inference_recognizer(model, 'demo/demo.mp4')
print(result)