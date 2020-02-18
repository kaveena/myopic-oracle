#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void reduce_nmckk_cpu(const int N, const int M, const int C, const int K, const Dtype * a, Dtype * y) {
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      Dtype accum = (Dtype) 0;
      for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
          accum += a[(n*M*C*K) + (m*C*K) + (c*K) + k];
        }
      }
      y[n*C +c] = accum;
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  if (this->saliency_term_) {
    ConvolutionSaliencyParameter conv_saliency_param = this->layer_param_.convolution_saliency_param();
    this->output_channel_saliency_compute_ = this->layer_param_.convolution_saliency_param().output_channel_compute();
    this->input_channel_saliency_compute_ = this->layer_param_.convolution_saliency_param().input_channel_compute();
    if (this->bias_term_ && this->layer_param_.convolution_saliency_param().saliency_bias()) {
      this->saliency_bias_ = true;
    }
    else {
      this->saliency_bias_ = false;
    }
    for (int i_s = 0; i_s < this->layer_param_.convolution_saliency_param().saliency_size(); i_s++){
      if (this->layer_param_.convolution_saliency_param().saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT) {
        this->separate_weight_diff_ = true;
      }
    }
  }
  else {
    this->separate_weight_diff_ = false;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->mask_term_ || this->quantize_term_) {
    weights_masked_shape_.clear();
    weights_masked_shape_.push_back(this->blobs_[this->mask_pos_]->count());
    weights_masked_.Reshape(weights_masked_shape_);
    if (this->bias_term_) {
      bias_masked_shape_.clear();
      bias_masked_shape_.push_back(this->blobs_[this->mask_pos_+1]->count());
      bias_masked_.Reshape(bias_masked_shape_);
    }
  }
  BaseConvolutionLayer<Dtype>::Reshape(bottom, top);
  this->compute_output_shape();
  if (this->saliency_term_) {
    output_saliencies_channel_.Reshape({(int)(this->layer_param_.convolution_saliency_param().saliency_size()), this->num_output_});
    output_saliencies_points_.Reshape(top[0]->shape()); //shape nmhw
    output_saliencies_filter_.Reshape({this->num_, this->num_output_}); //shape nm
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* bias;
  LayerParameter layer_param(this->layer_param_);
  if (layer_param.phase() == caffe::TRAIN) {
    this->quantize_clock_ += 1;
    this->activation_quantize_clock_ += 1;
  }
  if (this->quantize_term_) {
    Dtype* weight_masked = this->weights_masked_.mutable_cpu_data();
    caffe_and(this->blobs_[0]->count(), this->quantization_mask, weight, weight_masked);

    if (this->quantize_clock_ == this->quantize_interval_) {
      LOG(INFO) << "Quantizing weights";
      Dtype* weight_mut = this->blobs_[0]->mutable_cpu_data();
      caffe_copy(this->blobs_[0]->count(), weight_masked, weight_mut);
    }
    weight = this->weights_masked_.cpu_data();
  }
  if (this->mask_term_) {
    const Dtype* mask = this->blobs_[this->mask_pos_]->cpu_data();
    Dtype* weight_masked = this->weights_masked_.mutable_cpu_data();
    caffe_mul(this->blobs_[0]->count(), mask, weight, weight_masked);
    weight = this->weights_masked_.cpu_data();
  }
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
    if (this->quantize_term_) {
      Dtype* bias_masked = this->bias_masked_.mutable_cpu_data();
      caffe_and(this->blobs_[1]->count(), this->quantization_mask, bias, bias_masked);
      if (this->quantize_clock_ == this->quantize_interval_) {
        LOG(INFO) << "Quantizing biases";
        Dtype* bias_mut = this->blobs_[1]->mutable_cpu_data();
        caffe_copy(this->blobs_[1]->count(), bias_masked, bias_mut);
      }
      bias = this->bias_masked_.cpu_data();
    }
    if (this->mask_term_) {
      const Dtype* bias_mask = this->blobs_[this->mask_pos_+1]->cpu_data();
      Dtype* bias_masked = this->bias_masked_.mutable_cpu_data();
      caffe_mul(this->blobs_[1]->count(), bias_mask, bias, bias_masked);
      bias = this->bias_masked_.cpu_data();
    }
  }
  for (int i = 0; i < bottom.size(); ++i) {
    Dtype* bottom_data = bottom[i]->mutable_cpu_data();
    if (this->activation_quantize_term_ && (this->activation_quantize_clock_ == this->activation_quantize_interval_)) {
      caffe_and(bottom[i]->count(), this->activation_quantization_mask, bottom_data, bottom_data);
    }
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  if ((layer_param.phase() == caffe::TRAIN) && (this->quantize_clock_ >= this->quantize_interval_)) {
    this->quantize_clock_ = 0;
  }
  if ((layer_param.phase() == caffe::TRAIN) && (this->activation_quantize_clock_ >= this->activation_quantize_interval_)) {
    this->activation_quantize_clock_ = 0;
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* weights_sqr = this->weights_sqr_.mutable_cpu_data();
  Blob<Dtype>  weights_n_masked_;
  Blob<Dtype> bias_n_masked_;
  Blob<Dtype> input_shaped_blob_;
  Dtype* full_weights_diff;

  Dtype* weight_ddiff;
  Dtype* full_weights_ddiff;

  Dtype* bias_diff;
  Dtype* full_bias_diff;
  Dtype* bias_ddiff;
  Dtype* full_bias_ddiff;

  ConvolutionSaliencyParameter conv_saliency_param = this->layer_param_.convolution_saliency_param();

  if (this->saliency_term_ && this->separate_weight_diff_) {
    weights_n_masked_.Reshape({this->num_, this->blobs_[0]->shape()[0], this->blobs_[0]->shape()[1], this->blobs_[0]->shape()[2], this->blobs_[0]->shape()[3]});
    full_weights_diff = weights_n_masked_.mutable_cpu_diff();
  }

  if (this->mask_term_) {
    weight = this->weights_masked_.cpu_data();
  }

  if (this->saliency_term_) {
    if (Caffe::derivative_compute()) {
      weight_ddiff = this->blobs_[0]->mutable_cpu_diff();
      if (this->separate_weight_diff_) {
        full_weights_ddiff = weights_n_masked_.mutable_cpu_ddiff();
      }
    }
  }

  if (this->bias_term_) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();

    if (this->saliency_term_) {
      if (this->separate_weight_diff_) {
        bias_n_masked_.Reshape({this->num_, this->blobs_[1]->shape()[0]});
        full_bias_diff = bias_n_masked_.mutable_cpu_diff();
      }

      if (Caffe::derivative_compute()) {
        bias_ddiff = this->blobs_[1]->mutable_cpu_ddiff();
        if (this->separate_weight_diff_) {
          full_bias_ddiff = bias_n_masked_.mutable_cpu_ddiff();
        }
      }
    }
  }

  caffe_powx(this->blobs_[0]->count(), weight, (Dtype)2, weights_sqr);

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* top_data = top[i]->cpu_data();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    const Dtype* top_ddiff;
    Dtype* bottom_ddiff;
    Dtype* input_sqr_;
    if (Caffe::derivative_compute()) {
      input_shaped_blob_.Reshape(bottom[i]->shape());
      top_ddiff = top[i]->cpu_ddiff();
      bottom_ddiff = bottom[i]->mutable_cpu_ddiff();
      weight_ddiff = this->blobs_[0]->mutable_cpu_ddiff();
      if (this->separate_weight_diff_) {
        full_weights_ddiff = weights_n_masked_.mutable_cpu_ddiff();
      }
      input_sqr_ = input_shaped_blob_.mutable_cpu_data();
      caffe_powx(bottom[i]->count(), bottom[i]->cpu_data(), (Dtype) 2, input_sqr_);
    }
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
        if (this->separate_weight_diff_) {
          this->backward_cpu_bias_no_accum(full_bias_diff + n * this->blobs_[1]->count(), top_diff + n * this->top_dim_);
          caffe_add(this->blobs_[1]->count(), full_bias_diff + n * this->blobs_[1]->count(), bias_diff, bias_diff);
          if (Caffe::derivative_compute()) {
            this->backward_cpu_bias_no_accum(full_bias_ddiff + n * this->blobs_[1]->count(), top_ddiff + n * this->top_dim_);
            caffe_add(this->blobs_[1]->count(), full_bias_ddiff + n * this->blobs_[1]->count(), bias_ddiff, bias_ddiff);
          }
        }
        else {
          this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
          if (Caffe::derivative_compute()) {
            this->backward_cpu_bias(bias_ddiff, top_ddiff + n * this->top_dim_);
          }
        }
      }
      if (this->mask_term_) {
        caffe_mul(this->blobs_[1]->count(), this->blobs_[this->mask_pos_+1]->cpu_data(), bias_diff, bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          if (this->saliency_term_ && this->separate_weight_diff_) {
            this->weight_cpu_gemm_no_accum(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, full_weights_diff + n * this->blobs_[0]->count());
            caffe_add(this->blobs_[0]->count(), full_weights_diff + n * this->blobs_[0]->count(), weight_diff, weight_diff);
            if (Caffe::derivative_compute()) {
              this->weight_cpu_gemm_no_accum(input_sqr_ + n * this->bottom_dim_,
                  top_ddiff + n * this->top_dim_, full_weights_ddiff + n * this->blobs_[0]->count());
              caffe_add(this->blobs_[0]->count(), full_weights_ddiff + n * this->blobs_[0]->count(), weight_ddiff, weight_ddiff);
            }
          }
          else {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
            if (Caffe::derivative_compute()) {
              this->weight_cpu_gemm(input_sqr_ + n * this->bottom_dim_,
                top_ddiff + n * this->top_dim_, weight_ddiff);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
        if (Caffe::derivative_compute()) {
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_ddiff + n * this->top_dim_, weights_sqr,
                bottom_ddiff + n * this->bottom_dim_);
          }
        }
      }
      if (this->mask_term_) {
        // Don't update weights that are masked off
        caffe_mul(this->blobs_[0]->count(), this->blobs_[this->mask_pos_]->cpu_data(), weight_diff, weight_diff);
      }
    }

    // Compute Channel saliency
    // MULTIPLE INPUTS NOT TREATED
    if (this->saliency_term_) {
      Dtype * output_channel_saliency_data = NULL;
      Dtype * output_channel_saliency_accum_data = NULL;
      output_channel_saliency_data = output_saliencies_channel_.mutable_cpu_data();
      output_channel_saliency_accum_data = this->blobs_[this->saliency_pos_]->mutable_cpu_data();
      for (int i_s = 0; i_s < conv_saliency_param.saliency_size(); i_s++) {
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_taylor_cpu(top_data, top_diff, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_hessian_diag_cpu(top_data, top_ddiff, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_hessian_diag_approx2_cpu(top_data, top_diff, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_taylor_2nd_cpu(top_data, top_diff, top_ddiff, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_taylor_2nd_approx2_cpu(top_data, top_diff, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_INPUT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_weight_avg_cpu(top_data, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_GRADIENT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::ACTIVATION)){
            compute_diff_avg_cpu(top_diff, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_taylor_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_hessian_diag_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::HESSIAN_DIAG_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_hessian_diag_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX1) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_taylor_2nd_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::TAYLOR_2ND_APPROX2) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_taylor_2nd_approx2_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_INPUT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_weight_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
        if ((conv_saliency_param.saliency(i_s) == caffe::ConvolutionSaliencyParameter::AVERAGE_GRADIENT) && (conv_saliency_param.saliency_input(i_s) == caffe::ConvolutionSaliencyParameter::WEIGHT)){
          compute_diff_avg_weights_cpu(&weights_n_masked_, &bias_n_masked_, conv_saliency_param.saliency_norm(i_s), output_channel_saliency_data + (i_s * this->num_output_));
        }
      }
      if (this->layer_param_.convolution_saliency_param().accum()) {
        caffe_add(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data, output_channel_saliency_accum_data);
      }
      else {
        caffe_copy(output_saliencies_channel_.count(), output_channel_saliency_data, output_channel_saliency_accum_data);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_cpu(Dtype * output_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * output_channel_saliency) {

  int count = this->output_saliencies_points_.count(2,4);
  Dtype* filter_out_saliency_data = this->output_saliencies_filter_.mutable_cpu_data();
  switch (saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_abs(this->num_ * this->num_output_ * count, output_saliency_data, output_saliency_data);
      caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
  } break;

    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_powx(this->num_ * this->num_output_ * count, output_saliency_data, (Dtype) 2, output_saliency_data);
      caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    } break;

    case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
      caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
      caffe_abs(this->num_ * this->num_output_, filter_out_saliency_data, filter_out_saliency_data);
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    } break;

    case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
      caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
      caffe_powx(this->num_ * this->num_output_, filter_out_saliency_data, (Dtype) 2, filter_out_saliency_data);
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    } break;

    default: {
      caffe_sum(this->num_ * this->num_output_, count, output_saliency_data, filter_out_saliency_data); //sum hxw
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
    } break;
  }
  caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_norm_and_batch_avg_weights_cpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * output_channel_saliency) {

  Dtype* filter_out_saliency_data;

  int kernel_size = this->blobs_[0]->count(2,4);
  int weights_count = this->blobs_[0]->count();
  int bias_count;

  if (this->bias_term_) {
    bias_count = this->blobs_[1]->count();
  }

  switch (saliency_norm_) {
    case (caffe::ConvolutionSaliencyParameter::L1): {
      caffe_abs(this->num_ * weights_count, weight_saliency_data, weight_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_abs(this->num_ * bias_count, bias_saliency_data, bias_saliency_data);
     }
      filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();
      caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
      }
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
    } break;

    case (caffe::ConvolutionSaliencyParameter::L2): {
      caffe_powx(this->num_ * weights_count, weight_saliency_data, (Dtype) 2, weight_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_powx(this->num_ * bias_count, bias_saliency_data, (Dtype) 2, bias_saliency_data);
      }
      filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();
      caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
      }
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
    } break;

    case (caffe::ConvolutionSaliencyParameter::ABS_SUM): {
      filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();
      caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
      }
      caffe_abs(this->num_ * this->num_output_, filter_out_saliency_data, filter_out_saliency_data);
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
    } break;

    case (caffe::ConvolutionSaliencyParameter::SQR_SUM): {
      filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();
      caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
      }
      caffe_powx(this->num_ * this->num_output_, filter_out_saliency_data, (Dtype) 2, filter_out_saliency_data);
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
    } break;

    default: {
      filter_out_saliency_data = output_saliencies_filter_.mutable_cpu_data();
      caffe_sum(this->num_ * this->num_output_, this->channels_ * kernel_size / this->group_, weight_saliency_data, filter_out_saliency_data);
      if (this->saliency_bias_ && this->bias_term_ && bias_saliency_data != NULL){
        caffe_add(this->num_ * bias_count, bias_saliency_data, filter_out_saliency_data, filter_out_saliency_data);
      }
      caffe_strided_sum(this->num_output_, this->num_, filter_out_saliency_data, output_channel_saliency);
      caffe_scal(this->num_output_, (Dtype) 1.0 / (Dtype)(this->num_), output_channel_saliency);
    } break;
  }
}

template void ConvolutionLayer<float>::compute_norm_and_batch_avg_cpu(float * output_saliency_data, caffe::ConvolutionSaliencyParameter::NORM, float * output_channel_saliency);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_cpu(double * output_saliency_data, caffe::ConvolutionSaliencyParameter::NORM, double * output_channel_saliency);
template void ConvolutionLayer<float>::compute_norm_and_batch_avg_weights_cpu(float * weight_saliency_data, float * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM, float * output_saliency_data);
template void ConvolutionLayer<double>::compute_norm_and_batch_avg_weights_cpu(double * weight_saliency_data, double * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM, double * output_saliency_data);

#ifdef CPU_ONLY

template <typename Dtype>
void compute_taylor_gpu(const Dtype *  act_data, const Dtype *  act_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_,  Dtype * taylor_out) { NO_GPU; }
template <typename Dtype>
void compute_hessian_diag_gpu(const Dtype *  act_data, const Dtype *  act_ddiff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * hessian_diag_out) { NO_GPU; }
template <typename Dtype>
void compute_hessian_diag_approx2_gpu(const Dtype * act_data, const Dtype * act_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * hessian_diag_out) { NO_GPU; }
template <typename Dtype>
void compute_taylor_2nd_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_out) { NO_GPU; }
template <typename Dtype>
void compute_taylor_2nd_approx2_gpu(const Dtype *  act_data, const Dtype * act_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_out) { NO_GPU; }
template <typename Dtype>
void compute_weight_avg_gpu(const Dtype *  act_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_out) { NO_GPU; }
template <typename Dtype>
void compute_diff_avg_gpu(const Dtype *  act_diff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_out) { NO_GPU; }

template <typename Dtype>
void compute_taylor_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_out) { NO_GPU; }
template <typename Dtype>
void compute_hessian_diag_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * hessian_diag_out) { NO_GPU; }
template <typename Dtype>
void compute_hessian_diag_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * hessian_diag_out) { NO_GPU; }
template <typename Dtype>
void compute_taylor_2nd_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_out) { NO_GPU; }
template <typename Dtype>
void compute_taylor_2nd_approx2_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_out) { NO_GPU; }
template <typename Dtype>
void compute_weight_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_out) { NO_GPU; }
template <typename Dtype>
void compute_diff_avg_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * saliency_info_out) { NO_GPU; }

template <typename Dtype>
void compute_norm_and_batch_avg_gpu(Dtype * output_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * output_channel_saliency) { NO_GPU; }
template <typename Dtype>
void compute_norm_and_batch_avg_weights_gpu(Dtype * weight_saliency_data, Dtype * bias_saliency_data, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * output_channel_saliency) { NO_GPU; }

STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
