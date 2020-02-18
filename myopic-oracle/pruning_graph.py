import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf 
import numpy as np
import copy

def get_producer_convolution(net, initial_parents, producer_conv, conv_type, conv_index): 
  for i in range(len(initial_parents)): 
    initial_parent = initial_parents[i] 
    if '_split_' in initial_parent: 
      split_names = initial_parent.split('_split_') 
      initial_parent = split_names[0] + '_split' 
      conv_type.append('Split')
    if net.layer_dict[initial_parent].type == 'Concat':
      for j in range(len(net.bottom_names[initial_parent])):
        conv_type.append('Concat')
        conv_index.append(j)
    if ('Convolution' not in net.layer_dict[initial_parent].type) and ('Data' not in net.layer_dict[initial_parent].type) and ('InnerProduct' not in net.layer_dict[initial_parent].type): 
      get_producer_convolution(net, net.bottom_names[initial_parent], producer_conv, conv_type, conv_index) 
    else:
      producer_conv.append(initial_parent) 

def get_children(net, layer): 
  children = [] 
  if '_split_' in layer: 
    parent_split = layer.split('_split_')[0] + '_split' 
    next_layers = list(net.layer_dict.keys())[(list(net.layer_dict.keys()).index(parent_split)+1):] 
  else: 
    next_layers = list(net.layer_dict.keys())[(list(net.layer_dict.keys()).index(layer)+1):] 
  for each in next_layers: 
    if ((layer in net.bottom_names[each]) and (layer not in net.top_names[each])): 
      children.append(each) 
  if ((layer in net.top_names.keys() and net.layer_dict[layer].type == 'Split')): 
    children = list(net.top_names[layer]) 
  return children 

def get_consumer_convolution_or_fc(net, initial_consumers, consumer_sink, junction_type, conv_index, previous_layer): 
  for i in range(len(initial_consumers)): 
    initial_consumer = initial_consumers[i] 
    if '_split_' in initial_consumer: 
      split_names = initial_consumer.split('_split_') 
      parent_split = split_names[0] + '_split' 
      consumer_type = 'Split'
      junction_type.append(consumer_type)
      conv_index.append(split_names[1])
    else: 
      consumer_type = net.layer_dict[initial_consumer].type 
    if consumer_type == 'Concat': 
        junction_type.append('Concat') 
        conv_index.append(net.bottom_names[initial_consumer].index(previous_layer)) 
    if ('Convolution' not in consumer_type) and ('InnerProduct' not in consumer_type): 
      get_consumer_convolution_or_fc(net, get_children(net, initial_consumer), consumer_sink, junction_type, conv_index, initial_consumer) 
    else: 
      consumer_sink.append(initial_consumer) 

def get_sources(net, conv, producer_conv, conv_type, conv_index): 
  get_producer_convolution(net, net.bottom_names[conv], producer_conv, conv_type, conv_index) 

def get_sinks(net, conv, consumer_conv, conv_type, conv_index): 
  get_consumer_convolution_or_fc(net, get_children(net, conv), consumer_conv, conv_type, conv_index, conv) 

class PruningLayer:
  def __init__(self, prototxt_layer, caffe_layer):
    self.name = prototxt_layer.name
    self.caffe_layer = caffe_layer
    self.prototxt_layer = prototxt_layer
    self.type = prototxt_layer.type
    self.sources = []
    self.sinks = []
    self.source_junction_type = []
    self.sink_junction_type = []
    self.source_conv_index = []
    self.sink_conv_index = []
  def GetSources(self, caffe_net):
    get_sources(caffe_net, self.name, self.sources, self.source_junction_type, self.source_conv_index)
  def GetSinks(self, caffe_net):
    get_sinks(caffe_net, self.name, self.sinks, self.sink_junction_type, self.sink_conv_index)

class PruningConvolutionLayer(PruningLayer):
  def __init__(self, prototxt_layer, caffe_net, in_offset, out_offset):
    PruningLayer.__init__(self, prototxt_layer, caffe_net.layer_dict[prototxt_layer.name])
    self.GetSources(caffe_net)
    self.GetSinks(caffe_net)
    caffe_layer = caffe_net.layer_dict[self.name]
    self.bias_term = prototxt_layer.convolution_param.bias_term
    self.mask_term = prototxt_layer.convolution_mask_param.mask_term
    self.saliency_term = prototxt_layer.convolution_saliency_param.saliency_term
    self.output_channel_saliency_compute = prototxt_layer.convolution_saliency_param.output_channel_compute
    self.input_channel_saliency_compute = prototxt_layer.convolution_saliency_param.input_channel_compute
    mask_pos, saliency_pos = self.GetMaskAndSaliencyBlobPosition()
    self.mask_pos = mask_pos
    self.saliency_pos = saliency_pos
    # create helper attributes
    self.batch = int(caffe_net.blobs[self.name].num)
    self.height = int(caffe_net.blobs[self.name].height)
    self.width = int(caffe_net.blobs[self.name].width)
    self.output_channels = int(caffe_layer.blobs[0].data.shape[0])
    self.input_channels = int(caffe_layer.blobs[0].data.shape[1])
    self.kernel_size = int(caffe_layer.blobs[0].data.shape[2] * caffe_layer.blobs[0].data.shape[3])
    input_layer = caffe_net.bottom_names[self.name][0] 
    # create more helper attributes
    self.group = int(caffe_net.blobs[input_layer].channels / self.input_channels) 
    self.active_input_channels = np.ones(self.input_channels).astype(int)
    self.active_ifm = np.ones(self.input_channels * self.group).astype(int)
    self.active_output_channels = np.ones(self.output_channels).astype(int)
    self.active_ofm = np.ones(self.output_channels).astype(int)
    self.input_channel_idx = list(range(in_offset, in_offset + self.input_channels))
    self.output_channel_idx = list(range(out_offset, out_offset + self.output_channels))
    self.output_channel_input_idx = np.zeros(self.output_channels).astype(int).reshape(-1, 1).tolist()
    self.input_channel_output_idx = np.zeros(self.input_channels).astype(int).reshape(-1, 1).tolist()
  def GetMaskAndSaliencyBlobPosition(self):
    bias_pos = 1
    mask_pos = 1
    saliency_pos = 1
    if self.bias_term:
      mask_pos += 1
      saliency_pos += 1
    if self.mask_term:
      saliency_pos += 1
      if self.bias_term:
        saliency_pos += 1
    return mask_pos, saliency_pos
  def UpdateMaskInputChannel(self, idx_channel, final=True, fill=0):
    self.active_ifm[idx_channel] = 0 if fill == 0 else 1
    if final: 
      self.caffe_layer.blobs[0].data[:,idx_channel,:,:].fill(fill)
    if self.mask_term:
      self.caffe_layer.blobs[self.mask_pos].data[:,idx_channel,:,:].fill(0 if fill==0 else 1)
    self.active_input_channels[idx_channel] = 0 if fill == 0 else 1
    #print("pruned input channel ", idx_channel, self.name)
  def UpdateMaskOutputChannel(self, idx_channel, final=True, fill=0, prune_bias=True):
    if final:
      self.caffe_layer.blobs[0].data[idx_channel].fill(fill)
    if final and self.bias_term and prune_bias:
      self.caffe_layer.blobs[1].data[idx_channel] = fill
    self.caffe_layer.blobs[self.mask_pos].data[idx_channel].fill(0 if fill == 0 else 1)
    if self.bias_term and prune_bias:
      self.caffe_layer.blobs[self.mask_pos+1].data[idx_channel] = 0 if fill ==0 else 1
    self.active_output_channels[idx_channel] = 0 if fill == 0 else 1
    self.active_ofm[idx_channel] = 0 if fill == 0 else 1
    #print("pruned output channel ", idx_channel, self.name)

class PruningInnerProductLayer(PruningLayer):
  def __init__(self, prototxt_layer, caffe_net, in_offset, out_offset, input_channels=None, output_channels=None):
    PruningLayer.__init__(self, prototxt_layer, caffe_net.layer_dict[prototxt_layer.name])
    self.GetSources(caffe_net)
    self.GetSinks(caffe_net)
    caffe_layer = caffe_net.layer_dict[self.name]
    self.name = prototxt_layer.name
    self.bias_term = prototxt_layer.inner_product_param.bias_term
    # create helper attributes
    self.batch = int(caffe_net.blobs[self.name].num)
    self.height = int(caffe_net.blobs[self.name].height)
    self.width = int(caffe_net.blobs[self.name].width)
    self.input_channels = input_channels if input_channels is not None else int(caffe_layer.blobs[0].data.shape[1])
    self.output_channels = output_channels if output_channels is not None else int(caffe_layer.blobs[0].data.shape[0])
    self.output_size = int(caffe_layer.blobs[0].data.shape[0] / self.output_channels)
    self.input_size = int(caffe_layer.blobs[0].data.shape[1] / self.input_channels)
    # create more helper attributes
    self.active_input_channels = np.ones(self.input_channels).astype(int)
    self.active_ifm = np.ones(self.input_channels).astype(int)
    self.active_output_channels = np.ones(self.output_channels).astype(int)
    self.active_ofm = np.ones(self.output_channels).astype(int)
    self.input_channel_idx = list(range(in_offset, in_offset + self.input_channels))
    self.output_channel_idx = list(range(out_offset, out_offset + self.output_channels))
    self.output_channel_input_idx = np.zeros(self.output_channels).reshape(-1, 1).tolist()
    self.input_channel_output_idx = np.zeros(self.input_channels).reshape(-1, 1).tolist()
  def UpdateMaskInputChannel(self, idx_channel, final=True, fill=0):
    self.active_input_channels[idx_channel] = 0 if fill == 0 else 1
    self.active_ifm[idx_channel] = 0 if fill == 0 else 1
    if final:
      self.caffe_layer.blobs[0].data[:, int(idx_channel * self.input_size) : int((idx_channel * self.input_size) + self.input_size)].fill(fill)
    #print("pruned input channel ", idx_channel, self.name)

class PruningGraph:
  def __init__(self, caffe_net, prototxt_net):
    self.caffe_net = caffe_net
    self.prototxt = prototxt_net
    self.convolution_list = list(filter(lambda x: 'Convolution' in caffe_net.layer_dict[x].type, caffe_net.layer_dict.keys()))
    convolution_idx = list(filter(lambda x: 'Convolution' in prototxt_net.layer[x].type, range(len(prototxt_net.layer))))
    self.fc_list = list(filter(lambda x: 'InnerProduct' in caffe_net.layer_dict[x].type, caffe_net.layer_dict.keys()))
    fc_idx = list(filter(lambda x: 'InnerProduct' in prototxt_net.layer[x].type, range(len(prototxt_net.layer))))
    self.graph = dict()
    self.first_conv_layer = []
    self.last_conv_layer = []
    in_offset = 0
    out_offset = 0
    input_channels = []
    output_channels = []
    for prototxt_layer in convolution_idx:
      layer = prototxt_net.layer[prototxt_layer].name
      self.graph[layer] = PruningConvolutionLayer(prototxt_net.layer[prototxt_layer], caffe_net, in_offset, out_offset)
      in_offset += self.graph[layer].input_channels
      out_offset += self.graph[layer].output_channels
      output_channels.append(self.graph[layer].output_channels)
      input_channels.append(self.graph[layer].input_channels)
    for prototxt_layer in fc_idx:
      layer = prototxt_net.layer[prototxt_layer].name
      self.graph[layer] = PruningInnerProductLayer(prototxt_net.layer[prototxt_layer], caffe_net, in_offset, out_offset)
      if 'Convolution' in [caffe_net.layer_dict[x].type for x in self.graph[layer].sources]:
        self.graph[layer] = PruningInnerProductLayer(prototxt_net.layer[prototxt_layer], caffe_net, in_offset, out_offset, input_channels=int(self.graph[self.graph[layer].sources[0]].output_channels))
      in_offset += self.graph[layer].input_channels
      out_offset += self.graph[layer].output_channels
      output_channels.append(self.graph[layer].output_channels)
      input_channels.append(self.graph[layer].input_channels)
    self.input_channels_boundary = np.cumsum(np.array(input_channels))
    self.output_channels_boundary = np.cumsum(np.array(output_channels))
    self.first_conv_layer, self.last_conv_layer = self.GetFirstAndLastConvolution()
    self.total_output_channels = self.graph[self.last_conv_layer[-1]].output_channel_idx[-1] + 1
    self.total_input_channels = self.graph[self.last_conv_layer[-1]].output_channel_idx[-1] + 1
    self.UpdateChannelDependency()
  def GetFirstAndLastConvolution(self):
    first_layer = []
    last_layer = []
    for layer in self.convolution_list:
      conv_module = self.graph[layer]
      for l in conv_module.sinks:
        if 'Data' in self.caffe_net.layer_dict[l].type:
          first_layer.append(layer)
      for l in conv_module.sinks:
        if 'InnerProduct' in self.caffe_net.layer_dict[l].type:
          last_layer.append(layer)
          break
      if len(conv_module.sinks) == 0:
        last_layer.append(layer)
    return first_layer, last_layer
  def UpdateChannelDependency(self):
    for layer in self.convolution_list:
      for i in range(self.graph[layer].output_channels):
        self.graph[layer].output_channel_input_idx[i] = self.GetChannelOutputDependency(i, layer)
    for layer in self.convolution_list:
      for i in range(self.graph[layer].input_channels):
        self.graph[layer].input_channel_output_idx[i] = self.GetChannelInputDependency(i, layer)
    for layer in self.fc_list:
      for i in range(self.graph[layer].input_channels):
        self.graph[layer].input_channel_output_idx[i] = self.GetChannelInputDependency(i, layer)
  def GetChannelOutputDependency(self, idx_channel, idx_convolution):
    """
    Find which input channels are going to consume the output channel

    Parameters
    ----------
    idx_channel  : int
      Local index of output channel
    idx_convolution : string
      name of convolution channel
    """
    conv_module = self.graph[idx_convolution]
    sink_channel = list([])
    for i in range(len(conv_module.sinks)):
      c = conv_module.sinks[i]
      sink_module = self.graph[c]
      if sink_module.type == 'Convolution':
        if len(conv_module.sink_junction_type) >= len(conv_module.sinks):
          if conv_module.sink_junction_type[i] == 'Concat':
            c_offset = 0
            for j in range(len(sink_module.sources)):
              if j == conv_module.sink_conv_index[i]:
                break
              c_offset += self.graph[sink_module.sources[j]].output_channels
            sink_channel.append(sink_module.input_channel_idx[c_offset + idx_channel])
          if conv_module.sink_junction_type[i] == 'Split':
            sink_channel.append(sink_module.input_channel_idx[idx_channel])
        else:
          sink_channel.append(sink_module.input_channel_idx[idx_channel % sink_module.input_channels])
      else:
        sink_channel.append(sink_module.input_channel_idx[idx_channel])
    if len(conv_module.sinks) == 0:
      sink_channel = list([-1])
    return sink_channel
  def GetChannelInputDependency(self, idx_channel, idx_convolution):
    """
    Find which output channels are going to produce the input channel

    Parameters
    ----------
    idx_channel  : int
      Local index of input channel
    idx_convolution : string
      name of convolution channel
    """
    conv_module = self.graph[idx_convolution]
    source_channel = list([])
    global_input_idx = conv_module.input_channel_idx[idx_channel]
    if len(conv_module.sources) == 0:
      source_channel = list([-1])
    elif len(conv_module.source_junction_type) > 0:
      for s in conv_module.sources:
        src_module = self.graph[s]
        if conv_module.source_junction_type[0] == 'Concat':
          if global_input_idx in np.array(src_module.output_channel_input_idx).reshape(-1):
            if len(src_module.output_channel_input_idx[0]) == 1:
              local_output_channel = src_module.output_channel_input_idx.index([global_input_idx])
              source_channel.append(src_module.output_channel_idx[local_output_channel])
        if conv_module.source_junction_type[0] == 'Split':
          source_channel.append(src_module.output_channel_idx[idx_channel])
    elif conv_module.sources[0] in (self.convolution_list + self.fc_list):
      src_module = self.graph[conv_module.sources[0]]
      if src_module.type == 'Convolution':
        if conv_module.type == 'Convolution':
          for g in range(conv_module.group):
            source_channel.append(src_module.output_channel_idx[idx_channel + (g*conv_module.input_channels)])
        elif conv_module.type == 'InnerProduct':
          source_channel.append(src_module.output_channel_idx[idx_channel])
    else:
      source_channel = list([-1])
    return source_channel
  def GetChannelFromGlobalChannelIdx(self, global_idx, is_input_channel_idx=False):
    if is_input_channel_idx:
      channels = self.input_channels_boundary
    else:
      channels = self.output_channels_boundary
    layer_list = self.convolution_list + self.fc_list
    idx = np.where(channels>global_idx)[0][0]
    idx_convolution = layer_list[idx]
    idx_channel = (global_idx - channels[idx-1]) if idx > 0 else global_idx
    return idx_channel, idx_convolution
  def CheckLiveOutputChannel(self, global_channel_idx, skip_channels):
    channel, idx_convolution = self.GetChannelFromGlobalChannelIdx(global_channel_idx, False)
    live = False
    conv_module = self.graph[idx_convolution]
    for idx_sink in conv_module.output_channel_input_idx[channel]:
      if idx_sink not in skip_channels:
        i_c_sink, i_conv_sink = self.GetChannelFromGlobalChannelIdx(idx_sink, True)
        sink_module = self.graph[i_conv_sink]
        if sink_module.active_input_channels[i_c_sink] != 0:
          live = True
    return live
  def CheckLiveInputChannel(self, global_channel_idx, skip_channels):
    channel, idx_convolution = self.GetChannelFromGlobalChannelIdx(global_channel_idx, True)
    live = False
    conv_module = self.graph[idx_convolution]
    for idx_source in conv_module.input_channel_output_idx[channel]:
      if idx_source not in skip_channels:
        i_c_source, i_conv_source = self.GetChannelFromGlobalChannelIdx(idx_source, False)
        source_module = self.graph[i_conv_source]
        if source_module.active_output_channels[i_c_source] != 0:
          live = True
    return live
  def PruneChannel(self, pruned_channel, final=True, remove_all_nodes=False):
    """
    Prune a channel from the network.
    
    Parameters
    ----------
    pruned_channel: int
      global input or output index of the channel to remove
    final         : 
      if set to True, prunes a channel by also zeroing its weights
      if set to False, only the mask of the weights are zeroed out
    remove_all_nodes:
      set to True to remove dependency branch of that channel
      set to False to remove only pruned_channel and the weights that are inactive
      following the removal of pruned_channel
    """
    # find convolution and local channel index
    idx_channel, idx_convolution = self.GetChannelFromGlobalChannelIdx(pruned_channel, is_input_channel_idx=False)
    conv_module = self.graph[idx_convolution]
    # remove local weights
    conv_module.UpdateMaskOutputChannel(idx_channel, final)
    # update desencendants channels
    # if an output channel then update depencies of consumer channels
    if remove_all_nodes:
      found_sinks, found_sources = self.GetAllSinksSources(pruned_channel, False)
      for global_sink_input_idx in found_sinks:
        idx_sink, idx_conv_sink = self.GetChannelFromGlobalChannelIdx(global_sink_input_idx, True)
        self.graph[idx_conv_sink].UpdateMaskInputChannel(idx_sink, final)
      for global_source_output_idx in found_sources:
        idx_source, idx_conv_source = self.GetChannelFromGlobalChannelIdx(global_source_output_idx, False)
        self.graph[idx_conv_source].UpdateMaskOutputChannel(idx_source, final)
    for global_sink_input_idx in conv_module.output_channel_input_idx[int(idx_channel)]:
      if global_sink_input_idx > 0:
        live_channel = self.CheckLiveInputChannel(global_sink_input_idx, [pruned_channel])
        idx_sink, idx_conv_sink = self.GetChannelFromGlobalChannelIdx(global_sink_input_idx, True)
        if not live_channel:
          self.graph[idx_conv_sink].UpdateMaskInputChannel(idx_sink, final)
  def GetNumParam(self):
    conv_param = 0
    for l in self.convolution_list:
      layer = self.graph[l]
      conv_param += layer.kernel_size * layer.active_input_channels.sum() * layer.active_output_channels.sum()
    fc_param = 0
    for l in self.fc_list:
      layer = self.graph[l]
      fc_param += layer.active_input_channels.sum() * layer.input_size * layer.active_output_channels.sum() * layer.output_size
    return conv_param, fc_param
  def GenerateValidPrunePrototxt(self):
    proto_layers = [l.name for l in self.prototxt.layer]
    new_prototxt = copy.deepcopy(self.prototxt)
    new_weights = dict()
    for l in self.convolution_list:
      layer = self.graph[l]
      pruned_output_channels = np.nonzero(1-layer.active_output_channels)[0].tolist()
      pruned_input_channels = np.nonzero(1-layer.active_input_channels)[0].tolist()
      retained_output_channels = np.nonzero(layer.active_output_channels)[0].tolist()
      retained_input_channels = np.nonzero(layer.active_input_channels)[0].tolist()
      for local_output_idx in pruned_output_channels:
        if self.CheckGlobalLiveOutputChannel(layer.output_channel_idx[local_output_idx]):
          retained_output_channels += [local_output_idx]
      for local_input_idx in pruned_input_channels:
        if self.CheckGlobalLiveInputChannel(layer.input_channel_idx[local_input_idx]):
          retained_input_channels += [local_input_idx]
      retained_output_channels = np.sort(np.unique(retained_output_channels)).astype(int)
      retained_input_channels = np.sort(np.unique(retained_input_channels)).astype(int)
      # new prototxt params
      layer_idx = proto_layers.index(l)
      prototxt_layer = new_prototxt.layer[layer_idx]
      prototxt_layer.convolution_param.num_output = len(retained_output_channels)
      # remove additional param for mask blobs
      len_param = 1
      if prototxt_layer.convolution_param.bias_term:
        len_param = 2
      while len_param < len(prototxt_layer.param):
        prototxt_layer.param.pop()
      # remove mask and saliency params
      prototxt_layer.ClearField('convolution_saliency_param')
      prototxt_layer.ClearField('convolution_mask_param')
      # new weights
      new_conv_weights = layer.caffe_layer.blobs[0].data[:, retained_input_channels, :, :][retained_output_channels, :, :, :]
      if prototxt_layer.convolution_param.bias_term:
        new_conv_biases = layer.caffe_layer.blobs[1].data[retained_output_channels]
        new_weights[l] = [new_conv_weights, new_conv_biases]
      else:
        new_weights[l] = [new_conv_weights]
      if l + '_bn' in self.caffe_net.params.keys():
        bn_0 = self.caffe_net.layer_dict[l + '_bn'].blobs[0].data[retained_output_channels]
        bn_1 = self.caffe_net.layer_dict[l + '_bn'].blobs[1].data[retained_output_channels]
        bn_2 = self.caffe_net.layer_dict[l + '_bn'].blobs[2].data
        new_weights[l + '_bn'] = [bn_0, bn_1, bn_2]
      if l + '_scale' in self.caffe_net.params.keys():
        bn_0 = self.caffe_net.layer_dict[l + '_scale'].blobs[0].data[retained_output_channels]
        bn_1 = self.caffe_net.layer_dict[l + '_scale'].blobs[1].data[retained_output_channels]
        new_weights[l + '_scale'] = [bn_0, bn_1]
    for l in self.fc_list:
      layer = self.graph[l]
      retained_output_channels = np.nonzero(layer.active_output_channels)[0].tolist()
      retained_input_channels = np.nonzero(layer.active_input_channels)[0].tolist()
      retained_fc_out = [i_fc for c_fc in retained_output_channels for i_fc in range(layer.output_size*c_fc, layer.output_size*(c_fc + 1))]
      retained_fc_in = [i_fc for c_fc in retained_input_channels for i_fc in range(layer.input_size*c_fc, layer.input_size*(c_fc + 1))]
      retained_fc_out = np.sort(np.unique(retained_fc_out)).astype(int)
      retained_fc_in = np.sort(np.unique(retained_fc_in)).astype(int)
      # new prototxt params
      layer_idx = proto_layers.index(l)
      prototxt_layer = new_prototxt.layer[layer_idx]
      prototxt_layer.inner_product_param.num_output = len(retained_fc_out)
      # new weights
      new_fc_weights = layer.caffe_layer.blobs[0].data[retained_fc_out, :][:, retained_fc_in]
      if prototxt_layer.convolution_param.bias_term:
        new_fc_biases = layer.caffe_layer.blobs[1].data[retained_fc_out]
        new_weights[l] = [new_fc_weights, new_fc_biases]
      else:
        new_weights[l] = [new_fc_weights]
    return new_prototxt, new_weights
  def CheckGlobalLiveChannelItr(self, global_channel_idx, exclude_sinks, exclude_sources, live, is_input_channel=False):
    channel, idx_convolution = self.GetChannelFromGlobalChannelIdx(global_channel_idx, is_input_channel)
    if live:
      return True
    if is_input_channel and (global_channel_idx not in exclude_sinks):
      exclude_sinks.append(global_channel_idx)
      #print("checking input channel", global_channel_idx, idx_convolution, channel, exclude_sinks, exclude_sources)
      live = self.CheckLiveInputChannel(global_channel_idx, exclude_sources)
    elif not(is_input_channel) and (global_channel_idx not in exclude_sources):
      exclude_sources.append(global_channel_idx)
      #print("checking output channel", global_channel_idx, idx_convolution, channel, exclude_sinks, exclude_sources)
      live = self.CheckLiveOutputChannel(global_channel_idx, exclude_sinks)
    if live:
      return True
    if is_input_channel:
      list_sources = self.graph[idx_convolution].input_channel_output_idx[channel]
      for i_source in list(set(list_sources) - set(exclude_sources)):
        live = self.CheckGlobalLiveChannelItr(i_source, exclude_sinks, exclude_sources, live, is_input_channel=False)
    else:
      list_sinks = self.graph[idx_convolution].output_channel_input_idx[channel]
      for i_sink in list(set(list_sinks) - set(exclude_sinks)):
        live = self.CheckGlobalLiveChannelItr(i_sink, exclude_sinks, exclude_sources, live, is_input_channel=True)
    return live
  def GetAllSinksSourcesItr(self, global_channel_idx, found_sinks, found_sources, is_input_channel=False):
    channel, idx_convolution = self.GetChannelFromGlobalChannelIdx(global_channel_idx, is_input_channel)
    #print(found_sources, found_sinks)
    if is_input_channel:
      if global_channel_idx not in found_sinks:
        found_sinks.append(global_channel_idx)
        list_sources = self.graph[idx_convolution].input_channel_output_idx[channel]
        for i_source in list(set(list_sources) - set(found_sources)):
          self.GetAllSinksSourcesItr(i_source, found_sinks, found_sources, is_input_channel=False)
    else:
      if global_channel_idx not in found_sources:
        found_sources.append(global_channel_idx)
        list_sinks = self.graph[idx_convolution].output_channel_input_idx[channel]
        for i_sink in list(set(list_sinks) - set(found_sinks)):
          self.GetAllSinksSourcesItr(i_sink, found_sinks, found_sources, is_input_channel=True)
  def CheckGlobalLiveOutputChannel(self, channel):
    return self.CheckGlobalLiveChannelItr(channel, [], [], False, False)
  def CheckGlobalCanPruneOutputChannel(self, channel):
    return not(self.CheckGlobalLiveChannelItr(channel, [], [channel], False, False))
  def CheckGlobalLiveInputChannel(self, channel):
    return self.CheckGlobalLiveChannelItr(channel, [], [], False, True)
  def CheckGlobalCanPruneInputChannel(self, channel):
    return not(self.CheckGlobalLiveChannelItr(channel, [channel], [], False, True))
  def GetAllSinksSources(self, channel, is_input_channel=False):
    found_sinks = []
    found_sources = []
    self.GetAllSinksSourcesItr(channel, found_sinks, found_sources, is_input_channel)
    if is_input_channel:
      found_sinks.remove(channel)
    else:
      found_sources.remove(channel)
    return found_sinks, found_sources
  def LoadWeightsFromDict(self, weights):
    for l in weights.keys():
      for i in range(len(weights[l])):
        self.caffe_net.layer_dict[l].blobs[i].data[:] = weights[l][i]
  def UpdateActiveChannnels(self, ignore_bias=False):
    for l in self.convolution_list:
      for i in range(self.graph[l].output_channels):
        if not np.any(self.graph[l].caffe_layer.blobs[0].data[i]):
          if self.graph[l].bias_term and not(ignore_bias):
            if self.graph[l].caffe_layer.blobs[1].data[i] == 0:
              self.graph[l].active_output_channels[i] = 0
          else:
            self.graph[l].active_output_channels[i] = 0
      for i in range(self.graph[l].input_channels):
        if not np.any(self.graph[l].caffe_layer.blobs[0].data[:,i,:,:]):
          self.graph[l].active_input_channels[i] = 0
    for l in self.fc_list:
      for i in range(self.graph[l].output_channels):
        if not np.any(self.graph[l].caffe_layer.blobs[0].data[i*self.graph[l].output_size: (i+1)*self.graph[l].output_size - 1, :]):
          if self.graph[l].bias_term and not(ignore_bias):
            if self.graph[l].caffe_layer.blobs[1].data[i] == 0:
              self.graph[l].active_output_channels[i] = 0
          else:
            self.graph[l].active_output_channels[i] = 0
      for i in range(self.graph[l].input_channels):
        if not np.any(self.graph[l].caffe_layer.blobs[0].data[:, i*self.graph[l].input_size: (i+1)*self.graph[l].input_size - 1]):
          self.graph[l].active_input_channels[i] = 0
