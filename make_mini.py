import os
split2envs = {}
split2envs['train'] = ['17DRP5sb8fy', 
    'gZ6f7yhEvPG', 'dhjEzFoUFzH', 
    'PuKPg4mmafe', 'D7G3Y4RVNrH', 'Pm6F8kyY3z2', 'GdvgFV5R1Z5', 'YmJkqBEsHnH']
split2envs['val']  = ['pLe4wQe7qrG', 'oLBMNvg9in8']


for split, envs in split2envs.items():
    print(split, len(envs))
    if split=='val':
        root_dir = '/home/yutongchen/habitat-imitation-baselines/data/datasets/objectnav_mp3d_v1'
    else:
        root_dir = '/home/yutongchen/habitat-imitation-baselines/data/datasets/objectnav/objectnav_mp3d_70k'
    os.makedirs(os.path.join(root_dir,split+'_subset','content'), exist_ok=True)
    meta_src = os.path.join(root_dir,split,f"{split}.json.gz")
    meta_tgt = os.path.join(root_dir,split+'_subset', f"{split}_subset.json.gz")
    os.system(f'cp {meta_src} {meta_tgt}')
    for e in envs:
        env_dir_src= os.path.join(root_dir,split,'content',e+'.json.gz')
        env_dir_tgt= os.path.join(root_dir,split+'_subset','content',e+'.json.gz')
        os.system(f'cp -r {env_dir_src} {env_dir_tgt}')
