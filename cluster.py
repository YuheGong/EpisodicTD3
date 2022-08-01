import os

for i in range(5):
    str = f' python train.py --e Meta-context-dense-soccer-v2 --c 1 --seed {i}'
    #str = f' python train.py --e HopperXYJumpStepContext-v0 --c 1 --seed {i}'
    os.system(str)
    #str = f'python enjoy_1.py --env ALRReacherBalanceIP-v4 --seed {i}'
    #os.system(str)