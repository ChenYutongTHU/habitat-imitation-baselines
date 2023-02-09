'''
Adapted from 
https://github.com/Ram81/habitat-imitation-baselines/blob/d550cada6c71bc01b10253317281ffc5132ef5b2/habitat_baselines/il/disk_based/dataset/dataset.py
https://github.com/Ram81/habitat-imitation-baselines/blob/master/examples/objectnav_replay.py
#TODO 
1. Check the type of env, episodes, dataset
2. Check .config
3. Check the shape of "objectgoal":[], "compass(no)":[], "gps (no)":[] 
#run training to see (is there any other input batch_os?) [Align current inputs with env-based inputs]
4. 
'''
from typing import List
import lmdb, torch, os, json
import msgpack_numpy
import numpy as np
from torch.utils.data import Dataset
from habitat import logger
import habitat
from PIL import Image
from collections import defaultdict
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.utils.env_utils import make_env_fn
from habitat_baselines.common.environments import get_env_class

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self

def ObjectNavDisk_collate_fn(batch):
    '''
    Input: [{'observations':{'rgb':(D,), 'depth':(D,)}, 'demonstration':{'':}},...]
    Output:
    {'observations': {'rgb': (B,D),}, 'demonstration':{'action':(B,D)}}
    '''
    new_batch = {
        'idx':torch.tensor([b['idx'] for b in batch]), 
        'observations':[b['observations'] for b in batch],
        'demonstrations':[b['demonstrations'] for b in batch]}
    new_batch['observations'] = batch_obs(new_batch['observations'])
    new_batch['demonstrations'] = batch_obs(new_batch['demonstrations'])
    return  new_batch
'''
def collate_fn(batch):
    #TODO adapt for ObjectNav
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))
    
    observations_batch = list(transposed[1])
    next_actions_batch = list(transposed[2])
    prev_actions_batch = list(transposed[3])
    weights_batch = list(transposed[4])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )
        next_actions_batch[bid] = _pad_helper(next_actions_batch[bid], max_traj_len)
        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
    
    next_actions_batch = torch.stack(next_actions_batch, dim=1)
    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(next_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch,
        not_done_masks,
        next_actions_batch,
        weights_batch,
    )
'''

class ObjectNavDisk_Dataset(Dataset):
    """Pytorch dataset for object navigation task for the entire split"""

    def __init__(self, config, return_type, mode="train", use_iw=False, inflection_weight_coef=1.0, image_resize=None, image_preprocess=None):
        """
        Args:
            env (habitat.Env): Habitat environment
            config: Config
            return_type: 'raw'/'feature'
            mode: 'train'/'val'
        """
        #scene_split_name = mode
        # if content_scenes[0] != "*":
        #     scene_split_name = "_".join(content_scenes)
        self.config = config.TASK_CONFIG
        split = self.config.DATASET.SPLIT
        scene_split_name = mode
        self.return_type = return_type
        self.image_path, self.feature_path = {}, {}
        self.image_path['rgb'] = self.config.DATASET.DATA_PATH_DISK.IMAGE.RGB.format(split=split, scene_split=scene_split_name)
        self.image_path['depth'] = self.config.DATASET.DATA_PATH_DISK.IMAGE.DEPTH.format(split=split, scene_split=scene_split_name)
        self.feature_path['rgb'] = self.config.DATASET.DATA_PATH_DISK.FEATURE.RGB.format(split=split, scene_split=scene_split_name)
        self.feature_path['depth'] = self.config.DATASET.DATA_PATH_DISK.FEATURE.DEPTH.format(split=split, scene_split=scene_split_name)
        #self.label_path = config.DATASET.DATASET_PATH.LABEL.format(split=mode, scene_split=scene_split_name)
        
        self.config.defrost()
        #self.config.DATASET.CONTENT_SCENES = content_scenes #TODO  ? needed?
        self.config.freeze()
        self.image_resize = image_resize
        self.resolution = [self.config.SIMULATOR.RGB_SENSOR.WIDTH, self.config.SIMULATOR.RGB_SENSOR.HEIGHT]
        self.possible_actions = self.config.TASK.POSSIBLE_ACTIONS

        self.total_actions = 0
        self.inflections = 0
        self.inflection_weight_coef = self.config.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF
        self.image_preprocess = image_preprocess
        if use_iw:
            self.inflec_weight = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weight = torch.tensor([1.0, 1.0])

        #set-up dataset
        if self.return_type in ['image_rgb','image_depth']:
            self.image_type = self.return_type.split('_')[1]
            if not self.image_cache_exists(self.image_type):
                #self.env = habitat.Env(config=self.config)
                self.env = make_env_fn(config, get_env_class(config.ENV_NAME))
                self.episodes = self.env.episodes 
                logger.info(
                    f"Dataset cache not found. Saving {self.image_type} scene images"
                )
                logger.info(
                    "Number of {} episodes: {}".format(mode, len(self.episodes))
                )

                self.lmdb_env = lmdb.open(
                    self.image_path[self.image_type],
                    map_size=int(2e12),
                    writemap=True,
                )

                self.count, self.skip_eps = 0, 0
                self.eps_start_inds = []
                for ei in range(len(self.episodes)):#, key=lambda e:e.episode_id):
                    try:
                        self.init_observation = self.env.reset()
                        # from PIL import Image
                        # Image.fromarray(self.init_observation['rgb']).save('debug/0.png')
                        # print('Save as debug/0.png')
                        # import ipdb; ipdb.set_trace()
                    except:
                        logger.info(f'{self.env.current_episode.episode_id} is_thda={self.env.current_episode.is_thda} Skip')
                        self.skip_eps += 1
                        continue
                    episode = self.env.current_episode
                    state_index_queue = range(0, len(episode.reference_replay) - 1) #exclude the last one? (what will happen after STOP)
                    self.eps_start_inds.append(self.count)
                    self.save_frames(state_index_queue, episode)
                    logger.info(f'Finish episode {episode.episode_id} {len(self.eps_start_inds)}/{len(self.episodes)} step={len(state_index_queue)}')
                #logger.info("Inflection weight coef: {}, N: {}, nI: {}".format(self.total_actions / self.inflections, self.total_actions, self.inflections))
                logger.info(f"Image-{self.image_type} database ready! #episodes={len(self.eps_start_inds)} skip {self.skip_eps} eps")
                with open(self.image_path[self.image_type]+'.index', 'w') as f:
                    json.dump(self.eps_start_inds, f)
                with open(self.feature_path[self.image_type]+'.index', 'w') as f:
                    json.dump(self.eps_start_inds, f)
                self.env.close()
            else:
                logger.info("Dataset cache found.")
                self.lmdb_env = lmdb.open(
                    self.image_path[self.image_type],
                    readonly=True,
                    lock=False,
                )
            self.nitem_per_image = 2 #observation, demonstraion
            self.dataset_length = int(self.lmdb_env.begin().stat()["entries"] / self.nitem_per_image) 
            self.lmdb_env.close()
            self.lmdb_env = None

        elif self.return_type=='feature':
            #assert self.label_cache_exists() #saved along with image
            assert self.feature_cache_exists(), self.feature_path
            logger.info("Dataset cache found.")
            self.lmdb_env, self.dataset_length = {}, None
            self.nitem_per_image = 2 #observation, demonstraion
            self.eps_start_inds = None
            for k, fpath in self.feature_path.items():
                self.lmdb_env[k] = lmdb.open(
                    fpath,
                    readonly=True,
                    lock=False,
                )
                tmp = json.load(open(fpath+'.index','r'))
                if self.eps_start_inds==None:
                    self.eps_start_inds=tmp
                else:
                    assert self.eps_start_inds==tmp

                if self.dataset_length==None:
                    self.dataset_length = int((self.lmdb_env[k].begin().stat()["entries"]) / self.nitem_per_image) 
                else:
                    if not self.dataset_length==int((self.lmdb_env[k].begin().stat()["entries"]) / self.nitem_per_image):
                        logger.info(k, self.dataset_length, int(self.lmdb_env[k].begin().stat()["entries"] / self.nitem_per_image))
                        raise ValueError
                self.lmdb_env[k].close()
                logger.info(f'Load {fpath}, #episodes={len(self.eps_start_inds)}, #steps={self.dataset_length}')
            self.lmdb_env = None
        else:
            raise ValueError


    def image_cache_exists(self, key) -> bool:
        if os.path.exists(self.image_path[key]):
            if os.listdir(self.image_path[key]):
                return True
        else:
            os.makedirs(self.image_path[key])
        return False
    
    def feature_cache_exists(self) -> bool:
        for k, path in self.feature_path.items():
            if os.path.exists(path):
                if os.listdir(path):
                    return True
            else:
                os.makedirs(path)
            return False

    def label_cache_exists(self) -> bool:
        if os.path.exists(self.label_path):
            if os.listdir(self.label_path):
                return True
        else:
            os.makedirs(self.label_path)
        return False

    def read_lmdb_np_to_tensor(self, cursor, key, dtype):
        binary = cursor.get(key.encode())
        nparray = np.frombuffer(binary, dtype=dtype)
        tensor = torch.from_numpy(np.copy(nparray))
        return tensor

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):
        if self.return_type in ['image_depth', 'image_rgb']:
            if self.lmdb_env is None:
                self.lmdb_env = lmdb.open(
                    self.image_path[self.image_type], 
                    map_size=int(2e12),
                    writemap=True,
                )
                self.lmdb_txn = self.lmdb_env.begin()
                self.lmdb_cursor = self.lmdb_txn.cursor()
            
            height, width = int(self.resolution[0]), int(self.resolution[1])
            obs_idx = "{0:0=6d}_obs".format(idx)
            observations_binary = self.lmdb_cursor.get(obs_idx.encode())
            observations = msgpack_numpy.unpackb(observations_binary, raw=False)
            for k, v in observations.items():
                obs = np.array(v)
                if k==self.image_type and self.image_preprocess is not None:
                    obs = self.image_preprocess(Image.fromarray(obs))
                    obs = np.array(obs)
                observations[k] = torch.from_numpy(obs)  
            assert self.image_type in observations

            demo_idx = "{0:0=6d}_demo".format(idx)
            demonstrations_binary = self.lmdb_cursor.get(demo_idx.encode())
            demonstrations = msgpack_numpy.unpackb(demonstrations_binary, raw=False)
            for k, v in demonstrations.items():
                demo = np.array(v)
                if k=='inflection_weight':
                    demo = self.inflection_weight_coef if demo!=1.0 else 1.0
                demonstrations[k] = demo


            return {
                'idx': idx, 
                'observations': observations, 
                'demonstrations': demonstrations}

        elif self.return_type=='feature':
            if self.lmdb_env is None:
                self.lmdb_env, self.lmdb_txn, self.lmdb_cursor = {}, {}, {}
                for k, fpath in self.feature_path.items():
                    self.lmdb_env[k] = lmdb.open(
                        fpath, #TODO 
                        writemap=True,
                    )
                    self.lmdb_txn[k] = self.lmdb_env[k].begin()
                    self.lmdb_cursor[k] = self.lmdb_txn[k].cursor()
            observations = {}
            obs_idx = "{0:0=6d}_obs".format(idx)
            for k in ['rgb','depth']:
                obs_binary = self.lmdb_cursor[k].get(obs_idx.encode())
                obs = msgpack_numpy.unpackb(obs_binary, raw=False)
                observations = {**obs, **observations} #gps, compass in rgb are overwritten by those in depth
            # observations["rgb"] = self.read_lmdb_np_to_tensor(self.lmdb_cursor["rgb"], "{0:0=6d}_rgb_feature".format(idx), dtype="float32")
            # observations["depth"] = self.read_lmdb_np_to_tensor(self.lmdb_cursor["depth"], "{0:0=6d}_depth_feature".format(idx), dtype="float32")
            #TODO more observations: gps, compass, object goal

            #TODO demonstrations: action, weight
            demonstrations = {}
            demo_idx = "{0:0=6d}_demo".format(idx)
            demo_binary = self.lmdb_cursor['rgb'].get(demo_idx.encode()) #should be the same in rgb/depth
            demonstrations = msgpack_numpy.unpackb(demo_binary, raw=False)
            #demonstrations['inflection_weight'] = torch.where(demonstrations['inflection_weight'] != 1.0, self.inflection_weight_coef, 1.0)
            demonstrations['inflection_weight'] = self.inflection_weight_coef if demonstrations['inflection_weight']!=1.0 else 1.0
            return {'idx':idx, 'observations': observations, 'demonstrations': demonstrations}
        else:
            raise ValueError


    def save_frames(
        self, state_index_queue: List[int], episode #TODO 
    ) -> None:
        if len(state_index_queue)==0:
            return
        r"""
        Writes rgb, seg, depth frames to LMDB.
        """
        reference_replay = episode.reference_replay
        logger.info("Replay len: {}".format(len(reference_replay)))
        for si, state_index in enumerate(state_index_queue):
            action = self.possible_actions.index(reference_replay[state_index].action)
            if si==0 and action==0:
                observation = self.init_observation              
            else:
                observation = self.env.step(action=action)[0]

            next_state = reference_replay[state_index + 1]
            next_action = self.possible_actions.index(next_state.action)

            prev_state = reference_replay[state_index]
            prev_action = self.possible_actions.index(prev_state.action) #update, go to the next step

            demonstrations, observations = {}, {}
            observations[self.image_type] = observation[self.image_type]
            if self.image_resize is not None:
                from PIL import Image
                observations[self.image_type] = np.array((Image.fromarray(observations[self.image_type]).resize(self.image_resize)))
            for k in ["objectgoal","compass","gps"]:
                observations[k] = observation[k] #numpy
            demonstrations['prev_action'] = prev_action #int
            demonstrations['next_action'] = next_action #int
            #demonstrations['inflection_weight'] = observation['inflection_weight'] #float !!
            
            if si==0 or next_action!=prev_action:
                demonstrations['inflection_weight'] = self.inflection_weight_coef
            else:
                demonstrations['inflection_weight'] = 1
            #print(demonstrations['inflection_weight'],self.inflection_weight_coef)
            if si==(len(state_index_queue)-1) or next_action=='STOP':
                demonstrations['done'] = True 
            else:
                demonstrations['done'] = False

            sample_key = "{0:0=6d}".format(self.count) #(for one step rather than one episode!)
            with self.lmdb_env.begin(write=True) as txn:
                txn.put((sample_key + "_obs").encode(), msgpack_numpy.packb(observations, use_bin_type=True))
                txn.put((sample_key + "_demo").encode(), msgpack_numpy.packb(demonstrations, use_bin_type=True))
            
            self.count += 1 

            if demonstrations['done'] == True:
                break