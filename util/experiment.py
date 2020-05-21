import os
import time
from datetime import datetime

class Experiment:
    def __init__(self, name, dry=False):
        self.name = name
        self.dry = dry
        if not dry:
            for ds in os.walk('out'):
                if ds[0].split('/')[-1] == self.name:
                    ans = ''
                    while not (ans == 'y' or ans == 'n'):
                        ans = input('Experiment already exists, do you want to continue (y/n)? ')
                    if ans == 'n':
                        exit()
            
            self.out_path = 'out/{}'.format(name)
            if not os.path.exists(self.out_path):
                os.mkdir(self.out_path)
        
    def run(self, f, params={}, k=1, plotter=None, verbose=True):
        print('Starting experiment {}{}'.format(self.name, '(dry run)' if self.dry else ''))
        now = datetime.now()
        dt_start = now.strftime("%b-%d-%Y_%H:%M:%S")
                
        start = time.time()
        
        for i in range(k):
            print('Iteration {}/{}'.format(i + 1, k))
            data = f(**params)
            if not self.dry:
                with open('{}/data{}.txt'.format(self.out_path, '' if k < 2 else '_{}'.format(i)), 'w') as of:
                    for item in data:
                        of.write("%s\n" % item)
        
        end = time.time()
        duration = end - start
        
        now = datetime.now()
        dt_end = now.strftime("%b-%d-%Y_%H:%M:%S")
        
        if not self.dry:
            with open('{}/params.txt'.format(self.out_path), 'w') as of:
                of.write('name={}\n'.format(self.name))
                of.write('start_time={}\n'.format(dt_start))
                of.write('end_time={}\n'.format(dt_end))
                of.write('duration={:.4f}s\n'.format(duration))
                
                for key in params:
                    if isinstance(params[key], list):
                        i = 0
                        for elem in params[key]:
                            of.write('{}[{}]={}\n'.format(key, i, elem))
                            i += 1
                    else:
                        of.write('{}={}\n'.format(key, params[key]))
        

