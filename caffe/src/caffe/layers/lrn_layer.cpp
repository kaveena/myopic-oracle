#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    split_top_vec_.clear();
    split_top_vec_.push_back(&product_input_);
    split_top_vec_.push_back(&square_input_);
    LayerParameter split_param;
    split_layer_.reset(new SplitLayer<Dtype>(split_param));
    split_layer_->SetUp(bottom, split_top_vec_);
    // Set up square_layer_ to square the inputs.
    square_bottom_vec_.clear();
    square_top_vec_.clear();
    square_bottom_vec_.push_back(&square_input_);
    square_top_vec_.push_back(&square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(Dtype(2));
    square_layer_.reset(new PowerLayer<Dtype>(square_param));
    square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    pool_top_vec_.clear();
    pool_top_vec_.push_back(&pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->set_pad(pre_pad_);
    pool_param.mutable_pooling_param()->set_kernel_size(size_);
    pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
    pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
    // the sum of a squared neighborhood (the output of pool_layer_).
    power_top_vec_.clear();
    power_top_vec_.push_back(&power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-beta_);
    power_param.mutable_power_param()->set_scale(alpha_);
    power_param.mutable_power_param()->set_shift(Dtype(1));
    power_layer_.reset(new PowerLayer<Dtype>(power_param));
    power_layer_->SetUp(pool_top_vec_, power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    product_bottom_vec_.clear();
    product_bottom_vec_.push_back(&product_input_);
    product_bottom_vec_.push_back(&power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
    product_layer_->SetUp(product_bottom_vec_, top);
  }
  if (Caffe::derivative_compute()) {
    this->helper_.Reshape(top[0]->shape());
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(num_, channels_, height_, width_);
    scale_.Reshape(num_, channels_, height_, width_);
    if (Caffe::derivative_compute()) {
      helper_.Reshape(num_, channels_, height_, width_);
    }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    split_layer_->Reshape(bottom, split_top_vec_);
    square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
    pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
    power_layer_->Reshape(pool_top_vec_, power_top_vec_);
    product_layer_->Reshape(product_bottom_vec_, top);
    break;
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_cpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < scale_.count(); ++i) {
    scale_data[i] = k_;
  }
  Blob<Dtype> padded_square(1, channels_ + size_ - 1, height_, width_);
  Dtype* padded_square_data = padded_square.mutable_cpu_data();
  caffe_set(padded_square.count(), Dtype(0), padded_square_data);
  Dtype alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < num_; ++n) {
    // compute the padded square
    caffe_sqr(channels_ * height_ * width_,
        bottom_data + bottom[0]->offset(n),
        padded_square_data + padded_square.offset(0, pre_pad_));
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c),
          scale_data + scale_.offset(n, 0));
    }
    for (int c = 1; c < channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(height_ * width_,
          scale_data + scale_.offset(n, c - 1),
          scale_data + scale_.offset(n, c));
      // add head
      caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c + size_ - 1),
          scale_data + scale_.offset(n, c));
      // subtract tail
      caffe_axpy<Dtype>(height_ * width_, -alpha_over_size,
          padded_square_data + padded_square.offset(0, c - 1),
          scale_data + scale_.offset(n, c));
    }
  }

  // In the end, compute output
  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
  caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, square_top_vec_);
  pool_layer_->Forward(square_top_vec_, pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = scale_.cpu_data();

  const Dtype* top_ddiff;
  Dtype* bottom_ddiff;

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
  Blob<Dtype> accum_ratio(1, 1, height_, width_);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
  // We hack a little bit by using the diff() to store an additional result
  Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
  caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
  Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
  caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

  // go through individual data
  int inverse_pre_pad = size_ - (size_ + 1) / 2;
  for (int n = 0; n < num_; ++n) {
    int block_offset = scale_.offset(n);
    // first, compute diff_i * y_i / s_i
    caffe_mul<Dtype>(channels_ * height_ * width_,
        top_diff + block_offset, top_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    caffe_div<Dtype>(channels_ * height_ * width_,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
        scale_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    // Now, compute the accumulated ratios and the bottom diff
    caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
    for (int c = 0; c < size_ - 1; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
    for (int c = 0; c < channels_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
          accum_ratio_data);
      // compute bottom diff
      caffe_mul<Dtype>(height_ * width_,
          bottom_data + top[0]->offset(n, c),
          accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy<Dtype>(height_ * width_, -cache_ratio_value,
          accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c));
      caffe_axpy<Dtype>(height_ * width_, -1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
  }
  if (Caffe::derivative_compute()) {
    top_ddiff = top[0]->cpu_ddiff();
    bottom_ddiff = bottom[0]->mutable_cpu_ddiff();

    Dtype scale1 = Dtype(4. * (beta_ + 1) * beta_ * alpha_ * alpha_ / (size_ * size_));
    Dtype scale2 = Dtype(2. * alpha_ * beta_ / size_);
    Dtype scale3 = (Dtype) scale2 * scale2;

    Blob<Dtype> accum_ratio2(1, 1, height_, width_);
    Dtype* padded_ratio2_data = padded_ratio.mutable_cpu_diff();
    Dtype* padded_ratio3_data = padded_ratio.mutable_cpu_ddiff();
    Dtype* accum_ratio2_data = accum_ratio.mutable_cpu_ddiff();
    Dtype* accum_ratio3_data = accum_ratio2.mutable_cpu_diff();
    Dtype* accum_ratio3_times_bottom = accum_ratio2.mutable_cpu_ddiff();

    int count = bottom[0]->count();

    Dtype* helper_data_ = this->helper_.mutable_cpu_data();

    caffe_powx(count, scale_data, (Dtype)   - 2 * beta_, bottom_ddiff);
    caffe_mul(count, top_ddiff, bottom_ddiff, bottom_ddiff); // nijk ** -2*beta * d2E/dy2ijk

    caffe_div(count, top_data, scale_data, helper_data_); // yijk / nijk
    caffe_mul(count, top_diff, helper_data_, helper_data_); // yijk / nijk * dE/dyijk
    caffe_axpy(count, (Dtype) -2 * scale2, helper_data_, bottom_ddiff); // nijk ** -2*beta * d2E/dy2ijk +  -2 * scale2 yijk / nijk * dE/dyijk

    caffe_powx(count, top_data, (Dtype) 2,  helper_data_);
    caffe_div(count, helper_data_, scale_data, helper_data_); // yijk ** 2 / nijk
    caffe_mul(count, top_ddiff, helper_data_, helper_data_); //yijk **2 / nijk d2E/dy2ijk
    caffe_axpy(count, (Dtype) -2 * scale2, helper_data_, bottom_ddiff); // nijk ** -2*beta * d2E/dy2ijk +  -2 * scale2 yijk / nijk * dE/dyijk + -2 * scale2 yijk**2 / nijk d2E/dy2ijk

    // go through individual data
    int inverse_pre_pad = size_ - (size_ + 1) / 2;
    for (int n = 0; n < num_; ++n) {
      int block_offset = scale_.offset(n);
      // first, y_i / s_i
      caffe_div<Dtype>(channels_ * height_ * width_,
          top_data + block_offset, scale_data + block_offset,
          padded_ratio2_data + padded_ratio.offset(0, inverse_pre_pad));
      // y_i / (s_i * s_i)
      caffe_div<Dtype>(channels_ * height_ * width_,
          padded_ratio2_data + padded_ratio.offset(0, inverse_pre_pad), scale_data + block_offset,
          padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
      // (y_i / s_i)**2
      caffe_powx(channels_ * height_ * width_,
          padded_ratio2_data + padded_ratio.offset(0, inverse_pre_pad), (Dtype) 2,
          padded_ratio3_data + padded_ratio.offset(0, inverse_pre_pad));
      // diff_i * y_i / (s_i * s_i)
      caffe_mul<Dtype>(channels_ * height_ * width_,
          top_diff + block_offset, padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
          padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
      // diff_i * y_i / s_i
      caffe_mul<Dtype>(channels_ * height_ * width_,
          top_diff + block_offset, padded_ratio2_data + padded_ratio.offset(0, inverse_pre_pad),
          padded_ratio2_data + padded_ratio.offset(0, inverse_pre_pad));
      // ddiff_i * (y_i / s_i)**2
      caffe_mul<Dtype>(channels_ * height_ * width_,
          top_ddiff + block_offset, padded_ratio3_data + padded_ratio.offset(0, inverse_pre_pad),
          padded_ratio3_data + padded_ratio.offset(0, inverse_pre_pad));
      // Now, compute the accumulated ratios and the bottom diff
      caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
      caffe_set(accum_ratio.count(), Dtype(0), accum_ratio2_data);
      caffe_set(accum_ratio.count(), Dtype(0), accum_ratio3_data);
      for (int c = 0; c < size_ - 1; ++c) {
        caffe_axpy<Dtype>(height_ * width_, 1.,
            padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
        caffe_axpy<Dtype>(height_ * width_, 1.,
            padded_ratio2_data + padded_ratio.offset(0, c), accum_ratio2_data);
        caffe_axpy<Dtype>(height_ * width_, 1.,
            padded_ratio3_data + padded_ratio.offset(0, c), accum_ratio3_data);
      }
      for (int c = 0; c < channels_; ++c) {
        caffe_axpy<Dtype>(height_ * width_, 1.,
            padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
            accum_ratio_data);
        caffe_axpy<Dtype>(height_ * width_, 1.,
            padded_ratio2_data + padded_ratio.offset(0, c + size_ - 1),
            accum_ratio2_data);
        caffe_axpy<Dtype>(height_ * width_, 1.,
            padded_ratio3_data + padded_ratio.offset(0, c + size_ - 1),
            accum_ratio3_data);
        // compute bottom diff
        caffe_powx<Dtype>(height_ * width_,
            bottom_data + top[0]->offset(n, c),
            (Dtype) 2, accum_ratio3_times_bottom); // xijk **2
        caffe_mul<Dtype>(height_ * width_,
            accum_ratio3_times_bottom,
            accum_ratio_data, accum_ratio_times_bottom);

        caffe_mul<Dtype>(height_ * width_,
            accum_ratio3_times_bottom,
            accum_ratio3_data, accum_ratio3_times_bottom);

        caffe_axpy<Dtype>(height_ * width_, scale1,
            accum_ratio_times_bottom, bottom_ddiff + top[0]->offset(n, c));
        caffe_axpy<Dtype>(height_ * width_, - scale2,
            accum_ratio2_data, bottom_ddiff + top[0]->offset(n, c));
        caffe_axpy<Dtype>(height_ * width_, scale3,
            accum_ratio3_times_bottom, bottom_ddiff + top[0]->offset(n, c));

        caffe_axpy<Dtype>(height_ * width_, -1.,
            padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
        caffe_axpy<Dtype>(height_ * width_, -1.,
            padded_ratio2_data + padded_ratio.offset(0, c), accum_ratio2_data);
        caffe_axpy<Dtype>(height_ * width_, -1.,
            padded_ratio3_data + padded_ratio.offset(0, c), accum_ratio3_data);
      }
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    vector<bool> product_propagate_down(2, true);
    product_layer_->Backward(top, product_propagate_down, product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->Backward(square_top_vec_, propagate_down,
                            square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
    // the ddiff from split layer contains
    // nijk ** (-2 * beta )d2Edxijk
    //  + sum_u sum_v dE/dyi,j-u,k-v [ { 4 * (-beta ) (-beta - 1) xijk / n**2 } * xi,j-u,k-v ni,j-u,k-v ** (-beta - 2)
    //                                   + { -2 * beta * alpha / n } * xi,j-u,k-v ni,j-u,k-v ** (-beta - 1) ]
    //  + sum_u sum_v d2E/dy2i,j-u,k-v [ { 4 * beta **2 * alpha **2 xijk **2 / n**2} * xi,j-u,k-v ** 2 ni,j-u,k-v ** (-2beta -2)]

    // we need to add
    // -4 * beta * alpha / n  yijk nijk ** (- 1) dE/dyijk
    // -4 * beta * alpha / n yijk**2 nijk** (- 1) d2E/dy2ijk
    // nijk = ( k + alpha/n * sum_u sum_v xi,j-u,k-v **2 ) => use axpy on output of pool layer to get this
    if (Caffe::derivative_compute()) {
      int count = bottom[0]->count();
      Dtype* helper_data_ = this->helper_.mutable_cpu_data();
      Dtype* helper_data2_ = this->helper_.mutable_cpu_diff();

      caffe_axpy(count, this->alpha_, pool_top_vec_[0]->cpu_data(), helper_data_);
      caffe_add_scalar(count, this->k_, helper_data_); // nijk
      caffe_div(count, top[0]->cpu_data(), helper_data_, helper_data2_); // yijk / nijk
      caffe_mul(count, top[0]->cpu_diff(), helper_data2_, helper_data2_); // yijk / nijk dE/dyijk
      caffe_scal(count, (Dtype) -4 * this->beta_ * this->alpha_ / this->size_, helper_data2_); //-4 * beta * alpha / n *  yijk / nijk dE/dyijk
      caffe_add(count, helper_data2_, bottom[0]->cpu_ddiff(), bottom[0]->mutable_cpu_ddiff());

      caffe_div(count, top[0]->cpu_data(), helper_data_, helper_data2_); // yijk / nijk
      caffe_mul(count, top[0]->cpu_data(), helper_data2_, helper_data2_); // yijk**2 / nijk
      caffe_mul(count, top[0]->cpu_ddiff(), helper_data2_, helper_data2_); // yijk**2 / nijk d2E/dy2ijk
      caffe_scal(count, (Dtype) -4 * this->beta_ * this->alpha_ / this->size_, helper_data2_); //-4 * beta * alpha / n *  yijk**2 / nijk d2E/dy2ijk
      caffe_add(count, helper_data2_, bottom[0]->cpu_ddiff(), bottom[0]->mutable_cpu_ddiff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(LRNLayer);
STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNLayer, CrossChannelBackward);
STUB_GPU_BACKWARD(LRNLayer, WithinChannelBackward);
#endif

INSTANTIATE_CLASS(LRNLayer);

}  // namespace caffe
