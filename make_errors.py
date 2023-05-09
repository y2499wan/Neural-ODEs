import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='cifar')
# number of times each trail run
parser.add_argument('--nexps', type=int, default=3)
args = parser.parse_args()

models = ['node_'+args.dataset+'_v', 'sonode_conv_'+args.dataset+'_v', 'anode_'+args.dataset+'_v']
experiment_numbers = [str(i) for i in range(1,args.nexps+1)]
array_types = ['train_acc', 'test_acc']



def make_moving_av(model_no, experiment_no, array_type_no):

    model = models[model_no]
    experiment_no = experiment_numbers[experiment_no]
    array_type = array_types[array_type_no]
    
    filename = 'experiment_'+model+experiment_no+'/'
    
    accuracy = np.load(filename+array_type+'_arr.npy')
    samp_eps_array = np.load(filename+'epoch_arr.npy')
    
    
    window = 5
    def moving_average(a, periods=window):
        weights = np.ones(periods) / periods
        return np.convolve(a, weights, mode='valid')
    
    
    accuracy_ma = moving_average(accuracy)
    samp_eps_array = samp_eps_array[:len(samp_eps_array)-window+1]
    np.save(filename+'running_'+array_type+'.npy', accuracy_ma)
    np.save(filename+'running_epoch_arr.npy', samp_eps_array)


for i in range(len(models)):
    for j in range(len(experiment_numbers)):
        for k in range(len(array_types)):
            make_moving_av(i, j, k)