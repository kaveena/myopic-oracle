import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import argparse

colors = [
'#b2df8a',
'#d500ab',
'#00ffd5',
'#a6cee3',
'#1f78b4',
'#ffa2d1',
'#099981',
'#ff0000',
'#fdbf6f',
'#ff7f00',
'#d5d500',
'#6a3d9a',
'#b15928'
]

y_min=5
y_max=90

def parser():
    parser = argparse.ArgumentParser(description='Compare saliencies, augmented saliencies and hybrid oracles as pruning signals')
    parser.add_argument('--arch', action='store', default='LeNet-5',
            help='CNN model to use')
    parser.add_argument('--step-size', type=float, default=1, 
            help='Number of x axis points to use for interpolation and plotting')
    parser.add_argument('--k', type=int, default=16,
            help='number of channels for selection')
    parser.add_argument('--iterations', type=int, default=8,
            help='number of observations')
    parser.add_argument('--save', action='store_true', default=False,
            help='store plot')
    parser.add_argument('--oracle-eval-size', type=int, default=2,
            help='')
    parser.add_argument('--eval-size', type=int, default=2,
            help='')
    parser.add_argument('--accuracy-drop', type=float, default=5.0,
            help='')
    return parser

linestyle_dict = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),

     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 2, 1, 2)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 3, 1, 3, 1, 3)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

saliency_name = {
    'MEAN_ACTIVATIONS': 'Mean of activations (Equation 2)',
    '1ST_ORDER_TAYLOR': '1st order Taylor (Equation 5)',
    'FISHER_INFO': 'Fisher information (Equation 6)',
    'MEAN_GRADIENTS': 'Average of gradients (Equation 3)',
    'MEAN_SQR_WEIGHTS': 'Mean squares of weights (Equation 1)',
    'myopic_oracle' : 'Myopic oracle (our method)'
}

def get_linestyle(saliency):
  if saliency == "ACTIVATION-AVG-L1-l0_norm_adjusted":
    return linestyle_dict['dashdotted']
  elif saliency == "ACTIVATION-TAYLOR-ABS_SUM-l0_norm_adjusted":
    return 'dashed'
  elif saliency == "ACTIVATION-TAYLOR-SQR_SUM-no_normalisation":
    return 'dotted'
  elif saliency == "WEIGHT-AVG-L2-l0_norm_adjusted":
    return 'dashdot'
  elif saliency == "ACTIVATION-DIFF_AVG-ABS_SUM-l0_norm_adjusted":
    return linestyle_dict['dashdotdotted']
  else:
    return 'dashed'

args = parser().parse_args()

test_intervals = {'LeNet-5': 1, 'CIFAR10': 1, 'NIN': 10, 'ResNet-20': 1, 'AlexNet': 50}
test_interval = test_intervals[args.arch]
initial_test_accuracies = {'LeNet-5': 69.2, 'CIFAR10': 72.82, 'NIN': 88.18, 'ResNet-20': 88.39, 'AlexNet': 84.23}
initial_test_acc = initial_test_accuracies[args.arch]
savedir = args.arch

individual_saliencies='MEAN_ACTIVATIONS,1ST_ORDER_TAYLOR,FISHER_INFO,MEAN_GRADIENTS,MEAN_SQR_WEIGHTS'

saliencies = individual_saliencies.split(',')

x_signals =np.arange(0, 100 + args.step_size, args.step_size)

y_mean = np.zeros([len(saliencies),len(x_signals)])
y_sqr_mean = np.zeros([len(saliencies),len(x_signals)])
y_std = np.zeros([len(saliencies),len(x_signals)])

y_myopic_oracle_mean = np.zeros([len(x_signals)])
y_myopic_oracle_sqr_mean = np.zeros([len(x_signals)])
y_myopic_oracle_std = np.zeros([len(x_signals)])

iterations = args.iterations
eval_size = args.eval_size
oracle_eval_size = args.oracle_eval_size
k = args.k

# individual saliencies
for i_s in range(len(saliencies)):
    for j in range(1,iterations+1):
      dict_ = dict(np.load('/data/' + savedir + '/individual_saliencies/summary__' + saliencies[i_s] + '_evalsize' + str(eval_size) + '_iter' + str(j) + '.npy', allow_pickle=True).item())
      y_inter = scipy.interpolate.interp1d(np.hstack([0.0, 100.0 * (1-(dict_['conv_param'][::test_interval]/dict_['initial_conv_param'])), 100.0]), np.hstack([initial_test_acc, dict_['test_acc'][::test_interval], 10.0]))(x_signals)
      y_mean[i_s] += y_inter
      y_sqr_mean[i_s] += y_inter**2
y_mean = y_mean / iterations
y_sqr_mean = y_sqr_mean / iterations
y_std = (y_sqr_mean - (y_mean**2))
y_std = np.piecewise(y_std, [np.abs(y_std) < 1e-10, np.abs(y_std) > 1e-10], [0, lambda x: x])
y_std = y_std**0.5

# myopic oracle 
for j in range(1,iterations+1):
  dict_ = dict(np.load('/data/' + savedir + '/myopic_oracle/summary__MYOPIC_ORACLE_evalsize' + str(eval_size) + '_oracleevalsize' + str(oracle_eval_size) + '_k' + str(k) + '_iter' + str(j) + '.npy', allow_pickle=True).item())
  y_inter = scipy.interpolate.interp1d(np.hstack([0.0, 100.0 * (1-(dict_['conv_param'][::test_interval]/dict_['initial_conv_param'])), 100.0]), np.hstack([initial_test_acc, dict_['test_acc'][::test_interval], 10.0]))(x_signals)
  y_myopic_oracle_mean += y_inter
  y_myopic_oracle_sqr_mean += y_inter**2
y_myopic_oracle_mean = y_myopic_oracle_mean / iterations
y_myopic_oracle_sqr_mean = y_myopic_oracle_sqr_mean / iterations
y_myopic_oracle_std = (y_myopic_oracle_sqr_mean - (y_myopic_oracle_mean**2))**0.5

y_mean = np.piecewise(y_mean, [y_mean < 10, y_mean >= 10], [10, lambda x: x])
y_myopic_oracle_mean = np.piecewise(y_myopic_oracle_mean, [y_myopic_oracle_mean < 10, y_myopic_oracle_mean >= 10], [10, lambda x: x])


sparsity = x_signals
print(args.arch, initial_test_acc, dict_['initial_conv_param'])
for i_s in range(len(saliencies)):
  test_acc = y_mean[i_s]
  test_acc_b1 = y_mean[i_s] - 2*y_std[i_s, 0]
  test_acc_b2 = y_mean[i_s] + 2*y_std[i_s, 0]
  x_sparsity = scipy.interpolate.interp1d(test_acc, x_signals)(initial_test_acc - args.accuracy_drop)
  x_error_1_itr = scipy.interpolate.interp1d(test_acc_b1, x_signals)(initial_test_acc - args.accuracy_drop)
  x_error_2_itr = scipy.interpolate.interp1d(test_acc_b2, x_signals)(initial_test_acc - args.accuracy_drop)
  x_error = max(np.abs(x_error_1_itr - x_sparsity), np.abs(x_error_2_itr - x_sparsity))
  print(saliency_name[saliencies[i_s]], x_sparsity, '+-', x_error)
test_acc = y_myopic_oracle_mean
test_acc_b1 = y_myopic_oracle_mean - y_myopic_oracle_std
test_acc_b2 = y_myopic_oracle_mean + y_myopic_oracle_std
x_sparsity = scipy.interpolate.interp1d(test_acc, x_signals)(initial_test_acc - args.accuracy_drop)
x_error_1_itr = scipy.interpolate.interp1d(test_acc_b1, x_signals)(initial_test_acc - args.accuracy_drop)
x_error_2_itr = scipy.interpolate.interp1d(test_acc_b2, x_signals)(initial_test_acc - args.accuracy_drop)
x_error = max(np.abs(x_error_1_itr - x_sparsity), np.abs(x_error_2_itr - x_sparsity))
print(saliency_name["myopic_oracle"], x_sparsity, '+-', x_error)

fig = plt.figure(figsize=(27*0.4,18*0.4))
plot = fig.add_subplot(111)
for saliency in individual_saliencies:
  i_su = individual_saliencies.index(saliency)
  plt.plot(x_signals, y_mean[i_su], color=colors[i_su], label=saliency_name[saliency], linestyle=get_linestyle(saliency), linewidth=3.0)
  plt.fill_between(x_signals, y_mean[i_su] - 2*y_std[i_su], y_mean[i_su] + 2*y_std[i_su], color=colors[i_su], alpha=0.2)
plt.plot(x_signals, y_myopic_oracle_mean, color='k', label=saliency_name["myopic_oracle"] , linewidth=3.0)
plt.fill_between(x_signals, y_myopic_oracle_mean - 2*y_myopic_oracle_std, y_myopic_oracle_mean + 2*y_myopic_oracle_std, color='k', alpha=0.2)
plt.title(args.arch, fontsize=20)
plt.ylabel("Top-1 Test Accuracy", fontsize=20)
plt.xlabel("Convolution weights removed ($\%$)", fontsize=20)
plt.ylim((y_min, y_max))
if args.save:
  plt.savefig("/data/Graphs/" + args.arch + '-CIFAR10-results.pdf', bbox_inches='tight')
  figsize = (5, 1.5)
  fig_leg = plt.figure(figsize=figsize)
  ax_leg = fig_leg.add_subplot(111)
  ax_leg.legend(*plot.get_legend_handles_labels(), loc='center', ncol=3)
  ax_leg.axis('off')
  plt.savefig("/data/Graphs/" + args.arch + '-CIFAR10-legend.pdf', bbox_inches='tight')
else:
  plt.legend()
  plt.show()

