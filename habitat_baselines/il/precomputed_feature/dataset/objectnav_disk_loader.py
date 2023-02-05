import torch
from habitat_baselines.il.precomputed_feature.dataset.objectnav_disk import ObjectNavDisk_Dataset

class ObjectNavDisk_Sampler(torch.utils.data.Sampler):
    def __init__(self,dataset, generator):
        super().__init__(dataset)
        self.dataset = dataset
        self.generator = generator
        self.eps_num = len(self.dataset.eps_start_inds)
        self.set_epoch(epoch=0)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if epoch==0:
            self.eps_index = list(range(self.eps_num))
        else:
            self.eps_index = torch.randperm(self.eps_num, generator=self.generator).tolist()#np.random.permutation(self.eps_num)
        self.sample_index = []
        for ei in self.eps_index:
            start = self.dataset.eps_start_inds[ei]
            if ei==self.eps_num-1:
                end = len(self.dataset)
            else:
                end = self.dataset.eps_start_inds[ei+1]
            self.sample_index.extend(list(range(start, end)))

    def __len__(self):
        return len(self.dataset)
        
    def __iter__(self):
        pt = 0
        while True:
            yield self.sample_index[pt]
            pt += 1
            if pt==len(self.sample_index):
                self.set_epoch(self.epoch+1)
                pt = 0

def get_objectnav_loader(config, env_id=0):
    dataset = ObjectNavDisk_Dataset(
                config=config, return_type='feature', mode='train', 
                use_iw=config.IL.USE_IW, 
                inflection_weight_coef=config.TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF)
    generator = torch.Generator()
    generator.manual_seed(generator.initial_seed()+env_id)
    sampler = ObjectNavDisk_Sampler(dataset, generator=generator)
    sampler.set_epoch(env_id)
    loader =  torch.utils.data.DataLoader(dataset, batch_size=config.IL.BehaviorCloning.num_steps, 
                                           num_workers=config.NUM_WORKERS, sampler=sampler)
    return loader