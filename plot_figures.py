import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='cifar')
# number of times each trail run
parser.add_argument('--nexps', type=int, default=3)
args = parser.parse_args()


fig = plt.figure(figsize=[15, 4])
fig.subplots_adjust(hspace=0., wspace=0)
rc('font', family='serif')
rc('text', usetex=True)



######################################

sns.set_style('darkgrid')
ax1 = plt.subplot(1,3,1)


names = ['experiment_node_'+args.dataset+'_v', 'experiment_sonode_conv_'+args.dataset+'_v', 'experiment_anode_'+args.dataset+'_v']
labels =['NODE', 'SONODE', 'ANODE(1)']
colors = ['#DDAA33', '#004488', '#BB5566']
to_plot_names = ['epoch_arr.npy', 'running_train_acc.npy', 'running_test_acc.npy', 'nfe_b_arr.npy', 'nfe_f_arr.npy', 'time_avg_arr.npy', 'time_val_arr.npy']

def add_bit(x, to_plot, fname, ax):
    iters = np.load(names[x]+'1/'+ fname)
    loss_1 = np.load(names[x]+'1/'+to_plot_names[to_plot])
    # loss_2 = np.load(names[x]+'2/'+to_plot_names[to_plot])
    #loss_3 = np.load(names[x]+'3./'+to_plot_names[to_plot])
    
    loss = np.empty((len(loss_1),args.nexps))
    for i in range(len(loss_1)):
        loss[i][0] = loss_1[i]
    for j in range(1, args.nexps):
        doc = np.load(names[x]+str(j)+'/'+to_plot_names[to_plot])
        for i in range(len(loss_1)):
            loss[i][j] = doc[i]
    
    loss_mean = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_mean[i] = np.mean(loss[i])
    
    
    loss_std = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_std[i] = np.std(loss[i])
    
    print(loss_mean[len(loss_mean)-1])    
    print(loss_std[len(loss_std)-1]) 
    
    loss_p = loss_mean + loss_std
    loss_m = loss_mean - loss_std
    
    plt.plot(iters, loss_mean, color=colors[x], label=labels[x])
    ax.fill_between(x=iters, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])




add_bit(0, 1, 'running_epoch_arr.npy', ax1)    
add_bit(1, 1, 'running_epoch_arr.npy', ax1)
add_bit(2, 1, 'running_epoch_arr.npy', ax1)
rc('font', family='serif')
rc('text', usetex=True)
plt.legend(loc='lower right', fontsize=12)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Running Training Accuracy', fontsize=16)
plt.ylim(0.05, 0.99)
plt.title(f'{args.dataset.upper()} Training Accuracy', fontsize=22)


####################################################

ax2 = plt.subplot(1, 3, 2)

add_bit(0, 2, 'running_epoch_arr.npy', ax2)
add_bit(1, 2, 'running_epoch_arr.npy', ax2)
add_bit(2, 2, 'running_epoch_arr.npy', ax2)
rc('font', family='serif')
rc('text', usetex=True)
#plt.legend(loc='lower right', fontsize=12)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Running Test Accuracy', fontsize=16)
plt.ylim(0.05, 0.99)
plt.title(f'{args.dataset.upper()} Test Accuracy', fontsize=22)



##########################################################

ax3 = plt.subplot(1, 3, 3)

add_bit(0, 4, 'epoch_arr.npy', ax3)
add_bit(1, 4, 'epoch_arr.npy', ax3)
add_bit(2, 4, 'epoch_arr.npy', ax3)
rc('font', family='serif')
rc('text', usetex=True)
#plt.legend(fontsize=12)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('NFE', fontsize=16)
plt.ylim(10, 50)
plt.title(f'{args.dataset.upper()} NFE', fontsize=22)


plt.tight_layout()
plt.savefig(f'{args.dataset.upper()}.png', bbox_inches='tight')
