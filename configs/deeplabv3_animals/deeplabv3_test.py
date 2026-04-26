from mmseg.apis import init_model, inference_model
import mmcv
from mmseg.apis import MMSegInferencer
from mmseg.apis import show_result_pyplot
from pathlib import Path
# --- Настройки ---
config_path = '/home/ubuntu/animal_segmentation/configs/deeplabv3_animals/deeplabv3_train.py'
checkpoint_path = '/home/ubuntu/animal_segmentation/work_dir/iter_300.pth'
image_dir = '/home/ubuntu/animal_segmentation/animal_dataset/images'
my_classes = ['background', 'cat', 'dog']
my_palette = [
    [0, 0, 0],           # фон - чёрный
    [255, 0, 0],         # кошка - красный 
    [0, 255, 0]          # собака - зелёный
]
# ----------------
image_paths = [str(p) for p in Path(image_dir).glob('*.jpg')]
inferencer = MMSegInferencer(model=config_path, 
weights=checkpoint_path, 
classes=my_classes,
palette=my_palette,
device='cuda:0')
inferencer.model.dataset_meta = {
    'classes': my_classes,
    'palette': my_palette
}

# Запускаем пакетный инференс (сохраняет все результаты в папку)
inferencer(image_paths, out_dir="/home/ubuntu/animal_segmentation/inference_results", 
img_out_dir='/home/ubuntu/animal_segmentation/inference_results/img', 
pred_out_dir='/home/ubuntu/animal_segmentation/inference_results/masks',
opacity=0.5)