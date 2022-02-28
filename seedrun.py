import os

for i in range(5):
    str = f'python simple.py --e 3 --seed {i}'
    os.system(str)
    str = f'python simple.py --e 4 --seed {i}'
    os.system(str)
