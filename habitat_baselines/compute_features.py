import argparse, os
from tqdm import tqdm
import lmdb
import torch
import numpy as np
import msgpack_numpy
import habitat
from torch.utils.data import Dataloader
from habitat import Config, logger
from habitat_baselines.config.default import get_config
from habitat_baselines.il.common.encoders.backbone import VisualPretrainedEncoder
from habitat_baselines.il.precomputed_feature.dataset.objectnav_disk import ObjectNavDisk_Dataset, ObjectNavDisk_collate_fn

config = habitat.get_config("configs/tasks/objectnav_mp3d_il.yaml")
#B H W C -> B, 256
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_type", type=str, default="rgb"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="rgb"
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--output_log", type=str, default='compute_features_log.txt'
    )
    parser.add_argument(
        "--split", type=str, default='train_mini'
    )
    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    cfg = get_config(args.exp_config, opts)
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path

    os.makedirs(os.path.dirname(args.output_log), exists_ok=True)
    logger.add_filehandler(args.output_log)

    model = VisualPretrainedEncoder(image_type=args.image_type, model_cfg=cfg)
    model.cuda()
    
    dataset = ObjectNavDisk_Dataset(
        config=cfg, return_type=args.image_type, 
        mode=args.split, use_iw=False, inflection_weight_coef=1.0,
        image_preprocess = model.preprocess):
    dataloader = DataLoader(
        dataset,
        collate_fn=ObjectNavDisk_collate_fn,
        #collate_fn=collate_fn, (collate_fn is only needed in training)
        batch_size=args.batch_size,
        shuffle=False, ##!! We need to maintain the order
        num_workers=2,
        drop_last=False,
    )

    lmdb_env = lmdb.open(dataset.feature_path[args.image_type], writemap=True)
    lmdb_txn = lmdb_env.begin()
    logger.info('Save feature in '+dataset.feature_path[args.image_type])

    pbar = tqdm(total=len(dataloader))
    count = 0
    eps_start_inds = []
    with torch.no_grad():
        for batch in dataloader:
            #observations, demonstrations
            visual_input = batch['observations'][args.image_type].cuda()
            visual_feature = model(visual_input)
            pbar.update()
            for bi in range(visual_input.shape[0]):
                sample_key = "{0:0=6d}".format(count) 
                observations = {args.image_type: visual_feature[bi].cpu().numpy()}
                for obs_key, obs_value in batch['observations'].items():
                    if obs_key in observations:
                        continue #do not overwrite args.image_type
                    observations[obs_key] = obs_value[bi].cpu().numpy()
                demonstrations = {}
                for demo_key, demo_value in batch['demonstrations'].items():
                    demonstrations[demo_key] = demo_value[bi].cpu().numpy()              
                lmdb_txn.put((sample_key + f"_obs").encode(),msgpack_numpy.packb(observations, use_bin_type=True))
                lmdb_txn.put((sample_key + f"_demo").encode(),msgpack_numpy.packb(demonstrations, use_bin_type=True))
    lmdb_env.close()
    logger.info(f'Finish! #Total count={count}')


if __name__ == "__main__":
    main()