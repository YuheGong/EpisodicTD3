import os

for i in range(5):
    str = f'python enjoy_1.py --env ALRReacherBalanceIP-v3 --seed {i}'
    os.system(str)
    str = f'python enjoy_1.py --env ALRReacherBalanceIP-v4 --seed {i}'
    os.system(str)