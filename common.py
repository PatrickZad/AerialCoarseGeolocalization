import os
#os.environ['CUDA_VISIBLE_DEVICES']='2'
proj_path = os.getcwd()
print(proj_path)
base_dir = os.path.dirname(proj_path)
dataset_common_dir = os.path.join(base_dir, 'Datasets')
