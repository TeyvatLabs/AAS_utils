import requests, json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

class GenMP:

    def __init__(self, api_key):
        self.URL = 'http://localhost:5000'
        self.api_key = api_key

    def generate(self, src, config):
        param = {'src':src, 'config':config, 'api_key':self.api_key}
        res = requests.post(f'{self.URL}/generate', json=param)

        if res.status_code == 200:
            print('Processing ...')
            sec = 0
            task_id = res.json()['task_id']
            while sec < 600:
                try:
                    sec += 5
                    time.sleep(5)
                    result = requests.get(f'{self.URL}/result/{task_id}')
                    if result.json()['response']:
                        print('The model generation has been completed.')
                        return result.json()['payload']
                except:
                    sec += 5
                    time.sleep(5)
            return {'status': 'timed out'}
        else:
            return {'status': res.status_code}
    
    def decipher(self, src, model):
        param = {'src':src, 'model':model, 'api_key':self.api_key}
        res = requests.post(f'{self.URL}/decipher', json=param)
        
        if res.status_code == 200:
            return res.json()['payload']
        else:
            return {'status': res.status_code}

def simulate(src, mps, reach, display=True):
    if len(src.keys()) != len(mps.keys()):
        raise ValueError('ERROR: The period of input sources must match.')
    
    initials, src_T = ['o','h','l','c'], dict()
    for initial in initials:
        src_T[initial] = np.array([value[initials.index(initial)] for value in src.values()])
    
    mps = np.array(list(mps.values()))

    # Helper function to get label based on the given data
    def get_label(src_T, mps, reach):
        tmp_0 = np.where(np.logical_and(src_T['o'] < mps, mps < src_T['h']),
                         src_T['h']/mps-(1+reach),
                         np.nan)
        tmp_1 = np.where(tmp_0 > 0, 1, tmp_0)
        label = np.where(tmp_0 < 0, 0, tmp_1)
        return label

    tags = ['Period', 'Accuracy', 'Frequency', 'Batch Size', 'Batch Score Mean/Variance']

    # Overall simulation
    label = get_label(src_T, mps, reach)
    ps = np.count_nonzero(label==1)
    ls = np.count_nonzero(label==0)

    res = dict()
    res[tags[0]] = [list(src.keys())[0], list(src.keys())[-1], len(src.keys())]
    res[tags[1]] = round(ps/(ps+ls), 3)
    res[tags[2]] = round(((ps+ls)/label.shape[0]) * 100, 3)

    if display:
        print(f'{tags[0]}: {list(src.keys())[0]}~{list(src.keys())[-1]} ({len(src.keys())})\n')
        print(f'{tags[1]}: {res[tags[1]]}')
        print(f'{tags[2]}: {res[tags[2]]}%\n')

    # Calculate the batch scores
    batch_size = 50
    if label.shape[0] > 150:
        steps = range(0, label.shape[0], batch_size)
        batch = np.zeros((len(steps),), dtype='float')

        for idx in steps:
            extracted = label[idx:idx+batch_size]
            ps = np.count_nonzero(extracted==1)
            ls = np.count_nonzero(extracted==0)
            batch[int(idx/batch_size)] = ((ps-ls)*(ps+ls))/extracted.shape[0]

        if batch.var() > 0:
            score = batch.mean()/batch.var()
            res[tags[3]] = batch_size
            res[tags[4]] = round(score, 3)
        
        if display:
            print(f'{tags[3]}: {res[tags[3]]}')
            print(f'{tags[4]}: {res[tags[4]]}')

            # Draw a plot that represents the Score by Batch
            plt.plot(pd.Series(batch), color='black', linewidth=1)
            plt.axhline(y=0, color='red', linestyle='--', linewidth=0.5)

            plt.xticks(range(batch.shape[0]))

            plt.xlabel('Batch')
            plt.ylabel('Score')
            plt.show()
    
    # Construct the accuracy matrix by differnt ranges of target reach values (range: 0.0 to 0.1)
    matrix = dict()
    for r in range(100):
        label = get_label(src_T, mps, r/1000)
        ps = np.count_nonzero(label==1)
        ls = np.count_nonzero(label==0)
        matrix[r/1000] = ps/(ps+ls)

    if display:
        # Draw a plot that represents the Accuracy by Target Reach Value
        plt.plot(pd.Series(matrix), color='black', linewidth=1)
        plt.axhline(y=matrix[reach], color='red', linestyle='--', linewidth=0.5)

        plt.xlabel('Target Reach Value')
        plt.ylabel('Accuracy')
        plt.show()
    
    return res
