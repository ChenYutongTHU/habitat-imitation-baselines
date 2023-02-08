from typing import Optional, Dict, List
import torch
from gym import spaces
from habitat.core.spaces import ActionSpace
from habitat import Config, logger
from habitat_baselines.il.precomputed_feature.dataset.objectnav_disk_loader import get_objectnav_loader
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.utils.common import batch_obs
from collections import defaultdict
import gzip, json, numpy as np

def batch_envs(
    batches, 
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    new_batch: DefaultDict[str, List] = defaultdict(list)

    for b in batches:
        if type(b)==dict:
            for k,v in b.items(): #T,D
                new_batch[k].append(v) 

    batch_t: Dict[str, torch.Tensor] = {}

    for k in new_batch:
        batch_t[k] = torch.stack(new_batch[k], dim=1).to(device=device) #T,E,D

    return batch_t

class ObjectNavEnv_Offline(object):
    def __init__(self, config):
        self.config = config
        self.num_envs = config.NUM_PROCESSES

        self.dataloaders, self.observation_spaces, self.action_spaces = [], [], []
        self.category_to_task_category_id = json.load(
            gzip.open(self.config.TASK_CONFIG.DATASET.DATA_PATH.format(split=self.config.TASK_CONFIG.DATASET.SPLIT),'r'))['category_to_task_category_id']
        max_value = max(self.category_to_task_category_id.values())
        logger.info("max object cat: {}".format(max_value))
        logger.info("cats: {}".format(self.category_to_task_category_id.values()))
        for i in range(self.num_envs):
            logger.info(f'Set up env {i}/{self.num_envs} ...')
            loader = get_objectnav_loader(config=config, env_id=i)
            space_dict = {}
            for k, v in loader.dataset[0]['observations'].items():
                if k=='objectgoal':
                    space_dict[k] = spaces.Box(low=0, high=max_value, shape=v.shape, dtype=np.int64)
                else:
                    space_dict[k] = spaces.Space(shape=v.shape, dtype=v.dtype)
            self.observation_spaces.append(spaces.Dict(space_dict))
            self.dataloaders.append(iter(loader))
        self.num_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            #self.action_spaces.append(ActionSpace(self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))

    def next_batch(self, device):
        batch = {'observations':[], 'demonstrations':[]}
        for eid in range(self.num_envs):
            b = next(self.dataloaders[eid])
            batch['observations'].append(b['observations']) #T,D
            batch['demonstrations'].append(b['demonstrations'])
        batch['observations'] = batch_envs(batch['observations'], device) #T,E,D
        batch['demonstrations'] = batch_envs(batch['demonstrations'], device) #T,E,D
        return batch
    
    def close(self):
        return

