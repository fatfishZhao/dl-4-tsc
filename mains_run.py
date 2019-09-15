import os
import threading
import queue
import pandas as pd
import numpy as np
import datetime
from shutil import copyfile
import tqdm

class myThread (threading.Thread):
    def __init__(self,q, gpu_id):
        threading.Thread.__init__(self)
        self.q = q
        self.gpu_id = gpu_id
    def run(self):
        print ("开启GPU：" + str(self.gpu_id))
        process_data(self.q, self.gpu_id)
        print ("退出GPU：" + str(self.gpu_id))
def process_data(q, gpu_id):
    while 1:
        if not q.empty():
            data = q.get()
            if data=='break':
                break
            else:
                os.system(data+' '+str(gpu_id))
            print ("%s processing %s over" % (str(gpu_id), data))
start_i = 131
end_i = 141
datasets = ['AUSLAN', 'ArabicDigits', 'CMUsubject16', 'CharacterTrajectories', 'ECG', 'JapaneseVowels','KickvsPunch',
            'Libras', 'NetFlow', 'UWave', 'Wafer', 'WalkvsRun']
model_name = 'shuffle_torch'
workQueue = queue.Queue(12)
threads = []
for gpu_id in [0,2,3,4,5,6,7,8]:
    thread = myThread(workQueue, gpu_id)
    thread.start()
    threads.append(thread)
for dataset in datasets:
    for i in range(start_i,end_i):
        workQueue.put('/data2/zyx/software/anaconda3/bin/python main.py mts_archive '+dataset+ ' '+model_name+' _itr_'+str(i))
for i in range(8):
    workQueue.put('break')
print('game over')
for thread in threads:
    thread.join()
mean_list = []
std_list = []
for dataset in tqdm.tqdm(datasets):
    best_list = []
    for i in range(start_i,end_i):
        try:
            best_df = pd.read_csv(os.path.join('/data3/zyx/project/tsc/data/results/'+model_name, 'mts_archive_itr_'+str(i), dataset, 'df_metrics.csv'),
                                  index_col=0)
            best_df.index = range(best_df.shape[0])
            best_list.append(best_df['accuracy'][0])
        except:
            print('lost '+dataset+str(i))
    mean_list.append(np.mean(best_list))
    std_list.append(np.std(best_list))
result_df = pd.DataFrame({'dataset': datasets, 'mean':mean_list, 'std':std_list})
folder = './results/'+model_name+'/'+str(start_i)+'_'+str(end_i)+datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
os.makedirs(folder)
result_df.to_csv(os.path.join(folder, 'result.csv'))
copyfile('./utils/config.yaml', os.path.join(folder, 'config.yaml'))
