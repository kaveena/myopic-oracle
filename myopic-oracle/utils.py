import caffe
import os
import sys
import numpy as np
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

def get_pruning_signals(net, layer, saliencies):
  conv_module = net.graph[layer]
  pruning_signals = np.zeros((len(saliencies), conv_module.output_channels))
  for i_s in range(len(saliencies)):
    saliency = saliencies[i_s]
    weights = net.caffe_net.layer_dict[layer].blobs[0].data
    activations = net.caffe_net.blobs[layer].data
    gradients = net.caffe_net.blobs[layer].diff
    if saliency == 'MEAN_ACTIVATIONS':
      pruning_signals[i_s] = np.abs(activations).sum(axis=(0,2,3)) / float(activations.shape[0] * activations.shape[2] * activations.shape[3])
    elif saliency == '1ST_ORDER_TAYLOR':
      pruning_signals[i_s] = np.abs(activations*gradients).sum(axis=(0,2,3)) / float(activations.shape[0] * activations.shape[2] * activations.shape[3])
    elif saliency == 'FISHER_INFO':
      pruning_signals[i_s] = 0.5*((activations*gradients)**2).sum(axis=(0,2,3)) / float(activations.shape[0] * activations.shape[2] * activations.shape[3])
    elif saliency == 'MEAN_GRADIENTS':
      pruning_signals[i_s] = np.abs(gradients).sum(axis=(0,2,3)) / float(activations.shape[0] * activations.shape[2] * activations.shape[3])
    elif saliency == 'MEAN_SQR_WEIGHTS':
      pruning_signals[i_s] = (weights**2).sum(axis=(1,2,3)) / float(conv_module.active_input_channels.sum() * conv_module.kernel_size)
    else:
        print("Invalid saliency")
        sys.exit(1)
  return pruning_signals

def oracle(net, eval_channels, eval_size, remove_all_nodes=False):
  pruning_signal = -1 * np.ones(net.total_output_channels)
  for c in eval_channels:
    masked_sinks = []
    masked_sources = []
    found_sinks = []
    found_sources = []
    # Mask the channel
    idx_c, idx_conv = net.GetChannelFromGlobalChannelIdx(c, False)
    conv_module = net.graph[idx_conv]
    conv_module.UpdateMaskOutputChannel(idx_c, final=False, fill=0)
    # Mask its dependencies
    if remove_all_nodes:
      found_sinks, found_sources = net.GetAllSinksSources(c, False)
      for global_sink_input_idx in found_sinks:
        idx_sink, idx_conv_sink = net.GetChannelFromGlobalChannelIdx(global_sink_input_idx, True)
        if net.graph[idx_conv_sink].active_input_channels[idx_sink] == 1:
          masked_sinks.append(global_sink_input_idx)
          net.graph[idx_conv_sink].UpdateMaskInputChannel(idx_sink, final=False, fill=0)
      for global_source_output_idx in found_sources:
        idx_source, idx_conv_source = net.GetChannelFromGlobalChannelIdx(global_source_output_idx, False)
        if net.graph[idx_conv_source].active_output_channels[idx_source] == 1:
          masked_sources.append(global_source_output_idx)
          net.graph[idx_conv_source].UpdateMaskOutputChannel(idx_source, final=False, fill=0)
    eval_acc, eval_loss = test_network(net.caffe_net, eval_size)
    # Unmask the channel
    conv_module.UpdateMaskOutputChannel(idx_c, final=False, fill=1)
    # Unmask its dependencies
    if remove_all_nodes:
      for global_sink_input_idx in masked_sinks:
        idx_sink, idx_conv_sink = net.GetChannelFromGlobalChannelIdx(global_sink_input_idx, True)
        net.graph[idx_conv_sink].UpdateMaskInputChannel(idx_sink, final=False, fill=1)
      for global_source_output_idx in masked_sources:
        idx_source, idx_conv_source = net.GetChannelFromGlobalChannelIdx(global_source_output_idx, False)
        net.graph[idx_conv_source].UpdateMaskOutputChannel(idx_source, final=False, fill=1)
    pruning_signal[c] = eval_loss
  return pruning_signal

def get_k_channels(k, pruning_signals, active_channel):
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
    i_method = 0
    while(len(pruning_candidates) < k) and (i_k < len(active_channel) - 1):
      if sorted_channels[i_method][i_k] not in pruning_candidates:
        pruning_candidates = np.hstack([pruning_candidates, sorted_channels[i_method][i_k]])
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

