# Offline Training for Habitat-Imitation-Baselines
## Step 1. Extract RGB/Depth features
This will generate rendered RGB/depth images as intermediate outputs, which is saved as .lmdb for cache. The computed feature will also be saved in .lmdb form. Please refer to [objectnav_disk.py](habitat_baselines/il/precomputed_feature/dataset/objectnav_disk.py) for more details.
```bash
### Extract Depth features from resnet50.
path='habitat_baselines/config/objectnav/il_objectnav_disk.yaml'
image_type=depth
split=sample3
model_type=resnet50
python -m habitat_baselines.compute_features \
--exp-config  $path \
--image_type ${image_type} --output_log logs/compute_features/${split}/log.${image_type}_${model_type}.txt \
--split ${split} --batch_size 32 \
TASK_CONFIG.DATASET.SPLIT ${split} \
TASK_CONFIG.DATASET.DATA_PATH_DISK.IMAGE.DEPTH "/data-mount/datasets/objectnav/objectnav_mp3d_70k/{split}/{scene_split}.image_depth.db" \
TASK_CONFIG.DATASET.DATA_PATH_DISK.FEATURE.DEPTH "/data-mount/datasets/objectnav/objectnav_mp3d_70k/{split}/{scene_split}.depth_${model_type}.db" \
MODEL.DEPTH_ENCODER.TYPE ${model_type} 

### Extract RGB features from CLIP_ViT-B32/CLIP_RN50
path='habitat_baselines/config/objectnav/il_objectnav_disk.yaml'
image_type=rgb
split=sample3 #train_subset
model_type=CLIP_ViT-B32 # or CLIP_RN50
python -m habitat_baselines.compute_features \
--exp-config  $path \
--image_type ${image_type} --output_log logs/compute_features/${split}/log.${image_type}_${model_type}.txt \
--split ${split} --batch_size 32 \
TASK_CONFIG.DATASET.SPLIT ${split} \
TASK_CONFIG.DATASET.DATA_PATH_DISK.IMAGE.RGB "/data-mount/datasets/objectnav/objectnav_mp3d_70k/{split}/{scene_split}.rgb_image.db" \
TASK_CONFIG.DATASET.DATA_PATH_DISK.FEATURE.RGB "/data-mount/datasets/objectnav/objectnav_mp3d_70k/{split}/{scene_split}.rgb_${model_type}.db" \
MODEL.RGB_ENCODER.TYPE ${model_type}
```

## Step 2. Offline training using precomputed features
```
path='habitat_baselines/config/objectnav/il_objectnav_disk.yaml'
split=sample3
checkpoint_dir="/data-mount/results/train_subset/"

python -m habitat_baselines.run \
--exp-config  $path \
--run-type train \
TASK_CONFIG.DATASET.SPLIT ${split} \
TASK_CONFIG.DATASET.DATA_PATH data/datasets/objectnav/objectnav_mp3d_70k/{split}/{split}.json.gz \
TASK_CONFIG.DATASET.DATA_PATH_DISK.FEATURE.RGB "/data-mount/datasets/objectnav/objectnav_mp3d_70k/{split}/{split}.rgb_CLIP_RN50.db" \
TASK_CONFIG.DATASET.DATA_PATH_DISK.FEATURE.DEPTH "/data-mount/datasets/objectnav/objectnav_mp3d_70k/{split}/{split}.depth_resnet50.db" \
CHECKPOINT_FOLDER ${checkpoint_dir}
```

## Step 3. Online Inference 
```
path='habitat_baselines/config/objectnav/il_objectnav_disk.yaml'
spliteval=sample3
split=sample3
checkpoint="/data-mount/results/sample3/ckpt.9.pth"
output_dir="/data-mount/results/sample3_eval/"

python -m habitat_baselines.run \
--exp-config  $path \
--run-type eval \
TASK_CONFIG.DATASET.SPLIT ${spliteval} \
TASK_CONFIG.DATASET.DATA_PATH data/datasets/objectnav/objectnav_mp3d_70k/{split}/{split}.json.gz \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
EVAL_CKPT_PATH_DIR $checkpoint \
TEST_EPISODE_COUNT -1 \
EVAL.SPLIT ${split} \
TENSORBOARD_DIR ${output_dir}/tb \
VIDEO_DIR ${output_dir}/video \
OUTPUT_LOG_DIR ${output_dir}/log \
RESULTS_DIR ${output_dir}/results \
MODEL.RGB_ENCODER.TYPE CLIP_RN50 \
MODEL.DEPTH_ENCODER.TYPE resnet50
```
