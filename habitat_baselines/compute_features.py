import argparse, os
from tqdm import tqdm
import lmdb
import torch
import numpy as np
import msgpack_numpy
import habitat
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
    opts = vars(args)['opts']
    print(opts)
    cfg = get_config(args.exp_config, opts)
    cfg.defrost()
    cfg.freeze()
    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    logger.add_filehandler(args.output_log)

    model = VisualPretrainedEncoder(image_type=args.image_type, model_cfg=cfg)

    model.cuda()
    
    dataset = ObjectNavDisk_Dataset(
        config=cfg, return_type='image_'+args.image_type, 
        mode=args.split, use_iw=False, inflection_weight_coef=1.0,
        image_resize=(224,224) if args.image_type=='image' else None,
        image_preprocess = model.preprocess)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=ObjectNavDisk_collate_fn,
        #collate_fn=collate_fn, (collate_fn is only needed in training)
        batch_size=args.batch_size,
        shuffle=False, ##!! We need to maintain the order
        num_workers=2,
        drop_last=False,
    )

    lmdb_env = lmdb.open(dataset.feature_path[args.image_type], map_size=int(2e12), writemap=True)
    logger.info('Save feature in '+dataset.feature_path[args.image_type])

    pbar = tqdm(total=len(dataloader))
    count = 0
    eps_start_inds = []

    DEBUG, debug_save = True, []
    with torch.no_grad():
        for batch in dataloader:
            #observations, demonstrations
            debug_save.append(batch['observations'][args.image_type])
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
                with lmdb_env.begin(write=True) as txn:            
                    txn.put((sample_key + f"_obs").encode(),msgpack_numpy.packb(observations, use_bin_type=True))
                    txn.put((sample_key + f"_demo").encode(),msgpack_numpy.packb(demonstrations, use_bin_type=True))
                count += 1
    lmdb_env.close()
    logger.info(f'Finish! #Total count={count} visual_feature shape:{visual_feature[bi].cpu().numpy().shape}')
    torch.save(debug_save, 'debug_save.bin')

if __name__ == "__main__":
    #import ipdb; ipdb.set_trace()
    main()
