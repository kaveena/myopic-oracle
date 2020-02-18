import caffe
import os
import struct
import sys
import random
import numpy as np
from collections import Counter
import scipy
import scipy.stats
import copy
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf 

sys.dont_write_bytecode = True

def get_solver_proto_from_file(filename):
  solver_proto = caffe_pb2.SolverParameter()
  with open(filename) as f:
      s = f.read()
      txtf.Merge(s, solver_proto)
  return solver_proto

def get_prototxt_from_file(filename):
  prototxt_net = caffe_pb2.NetParameter()
  with open(filename) as f:
      s = f.read()
      txtf.Merge(s, prototxt_net)
  return prototxt_net

def save_prototxt_to_file(prototxt_net, filename):
  with open(filename, 'w') as f:
      f.write(str(prototxt_net))

def create_masked_prototxt(original_prototxt):
  for l in original_prototxt.layer:
    if l.type == 'Convolution':
      l.convolution_mask_param.Clear()
      l.convolution_mask_param.mask_term = True
      l.convolution_mask_param.mask_filler.type = 'constant'
      l.convolution_mask_param.mask_filler.value = 1
      expected_len_param = 1
      final_len_param = 2
      if l.convolution_param.bias_term:
        expected_len_param += 1
        final_len_param += 2
      current_len_param = len(l.param)
      for i in range(current_len_param, expected_len_param):
        l.param.add()
        l.param[i].lr_mult = 1
      current_len_param = len(l.param)
      for i in range(current_len_param, final_len_param):
        l.param.add()
        l.param[i].lr_mult = 0
        l.param[i].decay_mult = 0

def create_saliency_prototxt(original_prototxt, pointwise_saliency, saliency_input, saliency_reduction):
  if not (len(pointwise_saliency) == len(saliency_input) == len(saliency_reduction)):
    os.exit("Provided Saliencies Do Not Match")
  for l in original_prototxt.layer:
    if l.type == 'Convolution':
      l.convolution_saliency_param.Clear()
      l.convolution_saliency_param.saliency_term = True
      l.convolution_saliency_param.output_channel_compute = True
      l.convolution_saliency_param.accum = True
      for i in range(len(pointwise_saliency)):
        l.convolution_saliency_param.saliency.append(pointwise_saliency[i])
        l.convolution_saliency_param.saliency_input.append(saliency_input[i])
        l.convolution_saliency_param.saliency_norm.append(saliency_reduction[i])

def str2bool(v):
  """
    To parse bool from script argument
  """
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def saliency_scaling(net, layer, output_pruning_signal, scaling, input_saliency_type):
  """
  Apply scaling.
  
  Parameters
  ----------
  net         : caffe._caffe.Net
    CNN considered
  input_pruning_signal: array of floats
    saliency of input channels
  output_pruning_signal: array of floats
    saliency of output channels
  scaling     : type of scaling. options are
                  "l0_norm":  number of points used to compute channel saliency (does not consider updated/pruned network)
                  "l0_norm_adjusted":  number of points used to compute channel saliency (considering updated/pruned network)
                  "l1_norm":  layerwise L1-Norm of saliencies
                  "l2_norm":  layerwise L2-Norm of saliencies
                  "weights_removed":  number of weights to be removed if that channel is pruned
                  "no_scaling":  1.0
  """
    # saliency for output channels
  conv_module = net.graph[layer]
  if scaling == 'weights_removed':
    weights_removed = np.zeros(conv_module.output_channels)
    for c in range(conv_module.output_channels):
      weights_removed[c] = 0.0
      weights_removed[c] += conv_module.active_input_channels.sum() * conv_module.kernel_size
      if conv_module.bias_term:
        weights_removed[c] += 1
      for i_c in conv_module.output_channel_input_idx[c]:
        if i_c != -1:
          idx_channel, i_sink = net.GetChannelFromGlobalChannelIdx(i_c, True)
          sink = net.graph[i_sink]
          # if any add saliency of input channels
          if sink.type == 'Convolution':
            weights_removed[c] += sink.active_output_channels.sum() * sink.kernel_size
          if sink.type == 'InnerProduct':
            weights_removed[c] += sink.active_output_channels.sum() * sink.output_size * sink.input_size
        # if sink is not a convolution
    scale = np.piecewise(weights_removed, [weights_removed != 0.0, weights_removed == 0.0], [lambda x: x, 1.0])
  else:
    if scaling == 'l0_norm':
      if input_saliency_type == 'ACTIVATION':
        scale = conv_module.height * conv_module.width
      elif input_saliency_type == 'WEIGHT':
        scale = conv_module.input_channels * conv_module.kernel_size
    elif scaling == 'l0_norm_adjusted':
      if input_saliency_type == 'ACTIVATION':
        scale = conv_module.height * conv_module.width
      elif input_saliency_type == 'WEIGHT':
        scale = conv_module.active_input_channels.sum() * conv_module.kernel_size
    elif scaling == 'l1_norm':
      scale = np.abs(output_pruning_signal).sum()
    elif scaling == 'l2_norm':
      scale = (output_pruning_signal **2).sum()
    else:
      scale = 1.0
    if scale == 0.0:
      scale = 1.0
  return output_pruning_signal / scale

def test_network(net, itr):
  accuracy = dict()
  for i in range(itr):
    output = net.forward()
    for j in output.keys():
      if j in accuracy.keys():
        accuracy[j] = accuracy[j] + output[j]
      else:
        accuracy[j] = output[j].copy()
  for j in accuracy.keys():
    accuracy[j] /= float(itr)
  return accuracy['top-1']*100.0, accuracy['loss']

relu_pointwise = lambda x: x if x > 0.0 else 0.0
relu = np.vectorize(relu_pointwise)

def get_pruning_signals(net, layer, methods):
  pruning_signals = np.zeros((len(methods), net.graph[layer].output_channels))
  joint_methods = "-".join(methods)
  if 'ACTIVATION-TAYLOR' in joint_methods:
    activation_taylor = -1 * net.caffe_net.blobs[layer].data * net.caffe_net.blobs[layer].diff
  if 'ACTIVATION-TAYLOR_2ND_APPROX2' in joint_methods:
    activation_taylor_2nd_approx2 = (-1 * net.caffe_net.blobs[layer].data * net.caffe_net.blobs[layer].diff) + (0.5 * ((net.caffe_net.blobs[layer].data *net.caffe_net.blobs[layer].diff)**2))
  if 'WEIGHT-TAYLOR' in joint_methods:
    weight_taylor = -1 * net.caffe_net.layer_dict[layer].blobs[0].data * net.caffe_net.layer_dict[layer].blobs[0].diff
  if 'WEIGHT-TAYLOR_2ND_APPROX2' in joint_methods:
    weight_2nd_taylor_approx2 = (-1 * net.caffe_net.layer_dict[layer].blobs[0].data * net.caffe_net.layer_dict[layer].blobs[0].diff) + (0.5 * ((net.caffe_net.layer_dict[layer].blobs[0].data * net.caffe_net.layer_dict[layer].blobs[0].diff)**2))
  i = 0
  for method in methods:
    saliency_input, saliency_method, saliency_reduction, saliency_scaling = method.split('-')
    if saliency_input == 'ACTIVATION':
      if saliency_method == 'TAYLOR':
        pointwise_saliency = activation_taylor
      elif saliency_method == 'TAYLOR_2ND_APPROX2':
        pointwise_saliency = activation_taylor_2nd_approx2
      elif saliency_method == 'AVG':
        pointwise_saliency = net.caffe_net.blobs[layer].data
      elif saliency_method == 'AVG_RELU':
        pointwise_saliency = relu(net.caffe_net.blobs[layer].data)
      elif saliency_method == 'DIFF_AVG':
        pointwise_saliency = net.caffe_net.blobs[layer].diff
      elif saliency_method == 'HESSIAN_DIAG_APPROX2':
        pointwise_saliency = (0.5 * net.caffe_net.blobs[layer].data * net.caffe_net.blobs[layer].diff) ** 2
      elif saliency_method == 'apoz':
        pointwise_saliency = (net.caffe_net.blobs[layer].data > 0.0).astype(float) / (net.graph[layer].height * net.graph[layer].width)
      else:
        print("Invalid " + saliency_method + " pointwise saliency for activation based method")
        sys.exit(1)
      if saliency_reduction == 'L1':
        channel_saliency = np.abs(pointwise_saliency).sum(axis=(0,2,3))
      elif saliency_reduction == 'L2':
        channel_saliency = (pointwise_saliency **2).sum(axis=(0,2,3))
      elif saliency_reduction == 'ABS_SUM':
        channel_saliency = np.abs(pointwise_saliency.sum(axis=(2,3))).sum(axis=0)
      elif saliency_reduction == 'SQR_SUM':
        channel_saliency = (pointwise_saliency.sum(axis=(2,3))**2).sum(axis=0)
      elif saliency_reduction == 'NONE':
        channel_saliency = pointwise_saliency.sum(axis=(0,2,3))
      else:
        print("Saliency reduction invalid")
        sys.exit(1)
    if saliency_input == 'WEIGHT':
      if saliency_method == 'TAYLOR':
        pointwise_saliency = weight_taylor
      elif saliency_method == 'TAYLOR_2ND_APPROX2':
        pointwise_saliency = weight_2nd_taylor_approx2
      elif saliency_method == 'AVG':
        pointwise_saliency = net.caffe_net.layer_dict[layer].blobs[0].data
      elif saliency_method == 'DIFF_AVG':
        pointwise_saliency = net.caffe_net.layer_dict[layer].blobs[0].diff
      elif saliency_method == 'HESSIAN_DIAG_APPROX2':
        pointwise_saliency = (0.5 * net.caffe_net.layer_dict[layer].blobs[0].data * net.caffe_net.layer_dict[layer].blobs[0].diff) ** 2
      else:
        print("Invalid pointwise saliency for weight based method")
        exit(1)
      if saliency_reduction == 'L1':
        channel_saliency = np.abs(pointwise_saliency).sum(axis=(1,2,3))
      elif saliency_reduction == 'L2':
        channel_saliency = (pointwise_saliency **2).sum(axis=(1,2,3))
      elif saliency_reduction == 'ABS_SUM':
        channel_saliency = np.abs(pointwise_saliency.sum(axis=(2,3))).sum(axis=1)
      elif saliency_reduction == 'SQR_SUM':
        channel_saliency = (pointwise_saliency.sum(axis=(2,3))**2).sum(axis=1)
      elif saliency_reduction == 'NONE':
        channel_saliency = pointwise_saliency.sum(axis=(1,2,3))
      else:
        print("Saliency reduction invalid")
        exit(1)
    pruning_signals[i] = channel_saliency
    i = i + 1
  return pruning_signals

def oracle(net, eval_channels, eval_size):
  pruning_signal = -1 * np.ones(net.total_output_channels)
  for c in eval_channels:
    idx_c, idx_conv = net.GetChannelFromGlobalChannelIdx(c, False)
    conv_module = net.graph[idx_conv]
    if conv_module.bias_term:
      conv_module.caffe_layer.blobs[conv_module.mask_pos+1].data[idx_c] = 0
    conv_module.caffe_layer.blobs[conv_module.mask_pos].data[idx_c].fill(0)
    eval_acc, eval_loss = test_network(net.caffe_net, eval_size)
    if conv_module.bias_term:
      conv_module.caffe_layer.blobs[conv_module.mask_pos+1].data[idx_c] = 1
    conv_module.caffe_layer.blobs[conv_module.mask_pos].data[idx_c].fill(1)
    pruning_signal[c] = eval_loss
  return pruning_signal

def get_k_channels(k, pruning_signals, active_channel, use_random=False, exclude_channels=None):
  pruning_candidates = np.array([])
  candidate_channels = dict()
  sorted_channels = dict()
  num_methods = pruning_signals.shape[0]
  max_signals = num_methods
  i_k = 0
  if len(active_channel) > k:
    for i_method in range(num_methods):
      truncated_signal = pruning_signals[i_method][active_channel]
      sorted_channels[i_method] = np.array(active_channel)[np.argsort(truncated_signal).astype(int)]
      candidate_channels[i_method] = sorted_channels[i_method][:k]
      #print(candidate_channels[i_method])
    # take the set of channels chosen
    if use_random:
      random_channels = np.array(random.sample(active_channel, len(active_channel)))
      max_signals = num_methods + 1
    i_method = 0
    while(len(pruning_candidates) < k) and (i_k < len(active_channel) - 1):
      if i_method < num_methods:
        if sorted_channels[i_method][i_k] not in pruning_candidates:
          if exclude_channels is not None:
            if sorted_channels[i_method][i_k] not in exclude_channels:
              pruning_candidates = np.hstack([pruning_candidates, sorted_channels[i_method][i_k]])
          else:
            pruning_candidates = np.hstack([pruning_candidates, sorted_channels[i_method][i_k]])
      else:
        if random_channels[i_k] not in pruning_candidates:
          if exclude_channels is not None:
            if random_channels[i_k] not in exclude_channels:
              pruning_candidates = np.hstack([pruning_candidates, random_channels[i_k]])
          else:
            pruning_candidates = np.hstack([pruning_candidates, random_channels[i_k]])
      i_method = i_method + 1
      if i_method == max_signals:
        i_k = i_k + 1
        i_method = 0
  else:
    for i_method in range(num_methods):
        candidate_channels[i_method] = np.array(active_channel)
    pruning_candidates = np.array(active_channel)
  pruning_candidates = pruning_candidates.astype(int)
  return pruning_candidates, candidate_channels, i_k

def get_top_k_channels(k, pruning_signals, active_channel, use_random=False):
  pruning_candidates = np.array([])
  candidate_channels = dict()
  num_methods = pruning_signals.shape[0]
  if len(active_channel) > k:
    for i_method in range(num_methods):
      idx = np.argpartition(pruning_signals[i_method][active_channel], k)
      candidate_channels[i_method] = np.array(active_channel)[idx[:k]]
      #print(candidate_channels[i_method])
      pruning_candidates = np.hstack([pruning_candidates, candidate_channels[i_method]])
    # take the set of channels chosen
    if use_random:
      pruning_candidates = np.hstack([pruning_candidates, np.array(random.sample(active_channel, k))])
  else:
    for i_method in range(num_methods):
        candidate_channels[i_method] = np.array(active_channel)
    pruning_candidates = np.array(active_channel)
  pruning_candidates = np.unique(pruning_candidates.astype(int))
  return pruning_candidates, candidate_channels

def get_highest_voted_channel(k, pruning_signals, active_channel, total_channels):
  all_candidates = np.array([])
  all_candidates_ranks = np.array([])
  pruning_candidates = np.array([])
  candidate_channels = dict()
  num_methods = pruning_signals.shape[0]
  if len(active_channel) > k:
    for i_method in range(num_methods):
      idx = np.argpartition(pruning_signals[i_method][active_channel], k)
      candidate_channels[i_method] = np.array(active_channel)[idx[:k]].astype(int)
      #print(candidate_channels[i_method])
      candidate_channels_rank_cur = scipy.stats.rankdata(pruning_signals[i_method][candidate_channels[i_method]], method='min')
      all_candidates = np.hstack([all_candidates, candidate_channels[i_method]])
      all_candidates_ranks = np.hstack([all_candidates_ranks, candidate_channels_rank_cur])
    # take the set of channels chosen
    all_candidates = list(all_candidates)
    max_count = Counter(all_candidates).most_common()[0][1]
    pruning_candidates = [i for i in all_candidates if all_candidates.count(i)==max_count]
    pruning_candidates = np.unique(pruning_candidates).astype(int)
  else:
    for i_method in range(num_methods):
      candidate_channels[i_method] = np.array(active_channel)
      candidate_channels_rank_cur = scipy.stats.rankdata(pruning_signals[i_method][candidate_channels[i_method]], method='min')
      all_candidates_ranks = np.hstack([all_candidates_ranks, candidate_channels_rank_cur])
    pruning_candidates = np.array(active_channel)
  pruning_signal = np.zeros(total_channels)
  for i_channel in pruning_candidates:
    pruning_signal[i_channel] = all_candidates_ranks[np.where(all_candidates == i_channel)[0]].sum()
  return pruning_signal, pruning_candidates, candidate_channels

def get_rank(pruning_signals, active_channel, total_channels):
  pruning_candidates = np.array(active_channel)
  pruning_signal = np.zeros(total_channels)
  num_methods = pruning_signals.shape[0]
  for i_method in range(num_methods):
    pruning_signal[pruning_candidates] += scipy.stats.rankdata(pruning_signals[i_method][pruning_candidates], method='min')
  return pruning_signal, pruning_candidates
