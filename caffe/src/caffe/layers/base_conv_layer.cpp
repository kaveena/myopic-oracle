#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  ConvolutionMaskParameter conv_masked_param = this->layer_param_.convolution_mask_param();
  ConvolutionQuantizeParameter conv_quantized_param = this->layer_param_.convolution_quantize_param();
  ConvolutionSaliencyParameter conv_saliency_param = this->layer_param_.convolution_saliency_param();
  this->force_nd_im2col_ = conv_param.force_nd_im2col();
  this->channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = this->channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  this->num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(this->num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(this->num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  this->kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == this->num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < this->num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  this->stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = this->stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  this->pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = this->pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(this->num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << this->num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < this->num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  this->dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = this->dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << this->num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  this->is_1x1_ = true;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    this->is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!this->is_1x1_) { break; }
  }
  // Configure output channels and groups.
  this->channels_ = bottom[0]->shape(this->channel_axis_);
  this->num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(this->num_output_, 0);
  this->group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(this->channels_ % this->group_, 0);
  CHECK_EQ(this->num_output_ % this->group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    this->conv_out_channels_ = this->channels_;
    this->conv_in_channels_ = this->num_output_;
  } else {
    this->conv_out_channels_ = this->num_output_;
    this->conv_in_channels_ = this->channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  // - blobs_[2] holds the masks (optional)
  // - blobs_[3] holds the masks (optional)
  // - blobs_[4] holds the output channel saliencies (optional)
  // - blobs_[5] holds the input channel saliencies (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = this->conv_out_channels_;
  weight_shape[1] = this->conv_in_channels_ / this->group_;
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  this->bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(this->bias_term_, this->num_output_);
  this->mask_term_ = this->layer_param_.convolution_mask_param().mask_term();
  this->saliency_term_ = this->layer_param_.convolution_saliency_param().saliency_term();
  this->quantize_term_ = this->layer_param_.convolution_quantize_param().quantization_mode() != 0;
  this->quantize_interval_ = this->layer_param_.convolution_quantize_param().quantization_interval();
  this->quantize_clock_ = 0;
  if (this->quantize_term_) {
    int qrbits = this->layer_param_.convolution_quantize_param().quantization_range_bits();
    uint64_t qrbits_mask = (1 << qrbits) - 1;
    qrbits_mask <<= mantissa_length<Dtype>();
    int qpbits = this->layer_param_.convolution_quantize_param().quantization_precision_bits();
    uint64_t qpbits_mask = (1 << qpbits) - 1;
    qpbits_mask <<= (mantissa_length<Dtype>() - qpbits);
    uint64_t total_mask = qrbits_mask | qpbits_mask;
    this->quantization_mask = std::bitset<8*sizeof(Dtype)>(total_mask);
    this->quantization_mask[(8*sizeof(Dtype))-1] = true; // preserve sign bit
    LOG(INFO) << "Setting weight quantization scheme " << this->quantization_mask.to_string('*');
  }

  this->activation_quantize_term_ = this->layer_param_.activation_quantize_param().quantization_mode() != 0;
  this->activation_quantize_interval_ = this->layer_param_.activation_quantize_param().quantization_interval();
  this->activation_quantize_clock_ = 0;
  if (this->activation_quantize_term_) {
    int qrbits = this->layer_param_.activation_quantize_param().quantization_range_bits();
    uint64_t qrbits_mask = (1 << qrbits) - 1;
    qrbits_mask <<= mantissa_length<Dtype>();
    int qpbits = this->layer_param_.activation_quantize_param().quantization_precision_bits();
    uint64_t qpbits_mask = (1 << qpbits) - 1;
    qpbits_mask <<= (mantissa_length<Dtype>() - qpbits);
    uint64_t total_mask = qrbits_mask | qpbits_mask;
    this->activation_quantization_mask = std::bitset<8*sizeof(Dtype)>(total_mask);
    this->activation_quantization_mask[(8*sizeof(Dtype))-1] = true; // preserve sign bit
    LOG(INFO) << "Setting activation quantization scheme " << this->activation_quantization_mask.to_string('*');
  }
  int saliency_shape_0_ = 0;
  if (this->saliency_term_) {
    // check if the correct number of pointwise saliency, saliency input and norm have been provided
    if (!((conv_saliency_param.saliency_size() == conv_saliency_param.saliency_input_size()) 
      && (conv_saliency_param.saliency_size() == conv_saliency_param.saliency_norm_size()))) {
      LOG(FATAL) << "saliency, saliency_input and saliency_norm for each saliency measure" ;
    }
    saliency_shape_0_ = conv_saliency_param.saliency_size();
  }
  vector<int> saliency_out_shape = {saliency_shape_0_, this->num_output_};
  vector<int> saliency_in_shape = {saliency_shape_0_, this->channels_ / this->group_};
  int total_blobs = 1;
  this->mask_pos_ = 1;
  this->saliency_pos_ = 1;
  if (this->bias_term_) {
    total_blobs++;
    this->mask_pos_++;
    this->saliency_pos_++;
  }
  if ((this->mask_term_ || this->quantize_term_)) {
    total_blobs++;
    this->saliency_pos_++;
    if (this->bias_term_) {
      total_blobs++;
      this->saliency_pos_++;
    }
  }
  if (this->saliency_term_) {
    total_blobs++;
    total_blobs++;
  }
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + this->bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (this->bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
    if ((this->mask_term_ || this->quantize_term_) && weight_shape != this->blobs_[this->mask_pos_]->shape()) {
      Blob<Dtype> mask_shaped_blob(weight_shape);
      LOG(INFO) << "Incorrect mask shape: expected shape "
          << mask_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[this->mask_pos_]->shape_string();
      LOG(INFO) << "Mask Initialization: " << this->layer_param_.name();
      this->blobs_[this->mask_pos_].reset(new Blob<Dtype>(weight_shape));
      if (this->layer_param_.convolution_mask_param().default_init()) {
        Blob<Dtype> * mask_blob = this->blobs_[this->mask_pos_].get();
        for (int i=0; i<mask_blob->count(); ++i) {
          mask_blob->mutable_cpu_data()[i] = (Dtype)1.0;
        }
      }
      else {
        shared_ptr<Filler<Dtype> > mask_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_mask_param().mask_filler()));
        mask_filler->Fill(this->blobs_[this->mask_pos_].get());
      }
    }
    if (this->bias_term_ && (this->mask_term_ || this->quantize_term_) && bias_shape != this->blobs_[this->mask_pos_+1]->shape()) {
      Blob<Dtype> bias_mask_shaped_blob(bias_shape);
      LOG(INFO) << "Incorrect bias mask shape: expected shape "
          << bias_mask_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[this->mask_pos_+1]->shape_string();
      LOG(INFO) << "Mask Initialization: " << this->layer_param_.name();
      this->blobs_[this->mask_pos_+1].reset(new Blob<Dtype>(bias_shape));
      if (this->layer_param_.convolution_mask_param().default_init()) {
        Blob<Dtype> * mask_blob = this->blobs_[this->mask_pos_+1].get();
        for (int i=0; i<mask_blob->count(); ++i) {
          mask_blob->mutable_cpu_data()[i] = (Dtype)1.0;
        }
      }
      else {
        shared_ptr<Filler<Dtype> > mask_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_mask_param().mask_filler()));
        mask_filler->Fill(this->blobs_[this->mask_pos_+1].get());
      }
    }
    if (this->saliency_term_ && saliency_out_shape != this->blobs_[this->saliency_pos_]->shape()) {
      Blob<Dtype> saliency_out_shaped_blob(saliency_out_shape);
      LOG(FATAL) << "Incorrect saliency out shape: expected shape "
          << saliency_out_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[this->saliency_pos_]->shape_string();
      LOG(INFO) << "Saliency Initialization";
      this->blobs_[this->saliency_pos_].reset(new Blob<Dtype>(saliency_out_shape));
      Blob<Dtype> * saliency_out_blob = this->blobs_[this->saliency_pos_].get();
      for (int i=0; i<saliency_out_blob->count(); ++i) {
        saliency_out_blob->mutable_cpu_data()[i] = (Dtype)0.0;
      }
    }
    if (this->saliency_term_ && saliency_in_shape != this->blobs_[this->saliency_pos_+1]->shape()) {
      Blob<Dtype> saliency_in_shaped_blob(saliency_in_shape);
      LOG(FATAL) << "Incorrect saliency in shape: expected shape "
          << saliency_in_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[this->saliency_pos_+1]->shape_string();
      LOG(INFO) << "Saliency Initialization";
      this->blobs_[this->saliency_pos_+1].reset(new Blob<Dtype>(saliency_in_shape));
      Blob<Dtype> * saliency_in_blob = this->blobs_[this->saliency_pos_+1].get();
      for (int i=0; i<saliency_in_blob->count(); ++i) {
        saliency_in_blob->mutable_cpu_data()[i] = (Dtype)0.0;
      }
    }
  } else {
      this->blobs_.resize(total_blobs);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (this->bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    // If necessary, initialize and fill the mask.
    if ((this->mask_term_ || this->quantize_term_)) {
      this->blobs_[this->mask_pos_].reset(new Blob<Dtype>(weight_shape));
      if (this->layer_param_.convolution_mask_param().default_init()) {
        Blob<Dtype> * mask_blob = this->blobs_[this->mask_pos_].get();
        for (int i=0; i<mask_blob->count(); ++i) {
          mask_blob->mutable_cpu_data()[i] = (Dtype)1.0;
        }
      }
      else {
        shared_ptr<Filler<Dtype> > mask_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_mask_param().mask_filler()));
        mask_filler->Fill(this->blobs_[this->mask_pos_].get());
      }
    }
    if (this->bias_term_ && (this->mask_term_ || this->quantize_term_)) {
      this->blobs_[this->mask_pos_ +1 ].reset(new Blob<Dtype>(bias_shape));
      if (this->layer_param_.convolution_mask_param().default_init()) {
        Blob<Dtype> * mask_blob = this->blobs_[this->mask_pos_+1].get();
        for (int i=0; i<mask_blob->count(); ++i) {
          mask_blob->mutable_cpu_data()[i] = (Dtype)1.0;
        }
      }
      else {
        shared_ptr<Filler<Dtype> > mask_filler(GetFiller<Dtype>(
            this->layer_param_.convolution_mask_param().mask_filler()));
        mask_filler->Fill(this->blobs_[this->mask_pos_+1].get());
      }
    }
    if (this->saliency_term_) {
      this->blobs_[this->saliency_pos_].reset(new Blob<Dtype>(saliency_out_shape));
      Blob<Dtype> * saliency_out_blob = this->blobs_[this->saliency_pos_].get();
      for (int i=0; i<saliency_out_blob->count(); ++i) {
        saliency_out_blob->mutable_cpu_data()[i] = (Dtype)0.0;
      }
      this->blobs_[this->saliency_pos_+1].reset(new Blob<Dtype>(saliency_in_shape));
      Blob<Dtype> * saliency_in_blob = this->blobs_[this->saliency_pos_+1].get();
      for (int i=0; i<saliency_in_blob->count(); ++i) {
        saliency_in_blob->mutable_cpu_data()[i] = (Dtype)0.0;
      }
  }
  this->kernel_dim_ = this->blobs_[0]->count(1);
  this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / this->group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
        << " vs. bottom[" << bottom_id << "]: "
        << bottom[bottom_id]->shape_string();
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (this->bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  this->weights_sqr_.Reshape(this->blobs_[0]->shape());
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm_no_accum(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias_no_accum(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 0., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm_no_accum(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias_no_accum(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 0., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
