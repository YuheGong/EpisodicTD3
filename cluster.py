import os

for i in range(5):
    str = f'python train.py --env Meta-pick-place-v2 --seed {i}'
    os.system(str)
    #str = f'python enjoy_1.py --env ALRReacherBalanceIP-v4 --seed {i}'
    #os.system(str)