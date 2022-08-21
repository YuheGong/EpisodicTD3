import os

for i in range(5):
    #str = f'python train.py --e Meta-coffee-push-v2 --seed {i} --c 0'
    #os.system(str)
    str = f'python train.py --e Meta-dense-coffee-push-v2 --seed {i} --c 0'
    os.system(str)