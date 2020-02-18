#include <vector>

#include "caffe/layers/lrn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* const in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    const Dtype k, Dtype* const scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const in_off = in + offset;
    Dtype* const scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
                       * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}


template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* const in,
    const Dtype* const scale, const Dtype negative_beta, Dtype* const out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_, channels_, height_, width_, size_,
      alpha_ / size_, k_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
}
template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);


template <typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype cache_ratio, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    Dtype* const bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

template <typename Dtype>
__global__ void LRNComputeDdiff(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const top_data,
    const Dtype* const scale,
    const Dtype* const top_diff, const Dtype* const top_ddiff,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype negative_beta,
    const Dtype scale1, const Dtype scale2, const Dtype scale3,
    Dtype* const bottom_ddiff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    const Dtype* const bottom_off = bottom_data + offset;
    const Dtype* const top_off = top_data + offset;
    const Dtype* const scale_off = scale + offset;
    const Dtype* const top_diff_off = top_diff + offset;
    const Dtype* const top_ddiff_off = top_ddiff + offset;
    Dtype* const bottom_ddiff_off = bottom_ddiff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio1 = 0;
    Dtype accum_ratio2 = 0;
    Dtype accum_ratio3 = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio1 += top_diff_off[head * step] * top_off[head * step] /
          (scale_off[head * step] * scale_off[head * step]);
      accum_ratio2 += top_diff_off[head * step] * top_off[head * step] /
          (scale_off[head * step]);
      accum_ratio3 += top_ddiff_off[head * step] * pow(top_off[head * step] / scale_off[head * step], 2);
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio1 += top_diff_off[head * step] * top_off[head * step] /
          (scale_off[head * step] * scale_off[head * step]);
      accum_ratio2 += top_diff_off[head * step] * top_off[head * step] /
          (scale_off[head * step]);
      accum_ratio3 += top_ddiff_off[head * step] * pow(top_off[head * step] / scale_off[head * step], 2);
      if (head - size >= 0) {
        accum_ratio1 -= top_diff_off[(head - size) * step] * top_off[(head - size) * step] /
            (scale_off[(head - size) * step] * scale_off[(head - size) * step]);
        accum_ratio2 -= top_diff_off[(head - size) * step] * top_off[(head - size) * step] /
            (scale_off[(head - size) * step]);
        accum_ratio3 -= top_ddiff_off[(head - size) * step] * pow(top_off[(head - size) * step] / scale_off[(head - size) * step], 2);
      }
      bottom_ddiff_off[(head - post_pad) * step] =
        (pow(scale_off[(head - post_pad) * step], 2*negative_beta) * top_ddiff[(head - post_pad) * step])
        - (2 * scale2 * top_data[(head - post_pad) * step] * top_data[(head - post_pad) * step] * top_ddiff[(head - post_pad) * step] / scale_off[(head - post_pad) * step])
        - (2 * scale2 * top_data[(head - post_pad) * step] * top_diff[(head - post_pad) * step] / scale_off[(head - post_pad) * step])
        + (scale1 * bottom_off[(head - post_pad) * step] * bottom_off[(head - post_pad) * step] * accum_ratio1)
        - (scale2 * accum_ratio2)
        + (scale3 * bottom_off[(head - post_pad) * step] * bottom_off[(head - post_pad) * step] * accum_ratio3);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio1 -= top_diff_off[(head - size) * step] * top_off[(head - size) * step] /
            (scale_off[(head - size) * step] * scale_off[(head - size) * step]);
        accum_ratio2 -= top_diff_off[(head - size) * step] * top_off[(head - size) * step] /
            (scale_off[(head - size) * step]);
        accum_ratio3 -= top_ddiff_off[(head - size) * step] * pow(top_off[(head - size) * step] / scale_off[(head - size) * step], 2);
      }
      bottom_ddiff_off[(head - post_pad) * step] =
        (pow(scale_off[(head - post_pad) * step], 2*negative_beta) * top_ddiff[(head - post_pad) * step])
        - (2 * scale2 * top_data[(head - post_pad) * step] * top_data[(head - post_pad) * step] * top_ddiff[(head - post_pad) * step] / scale_off[(head - post_pad) * step])
        - (2 * scale2 * top_data[(head - post_pad) * step] * top_diff[(head - post_pad) * step] / scale_off[(head - post_pad) * step])
        + (scale1 * bottom_off[(head - post_pad) * step] * bottom_off[(head - post_pad) * step] * accum_ratio1)
        - (scale2 * accum_ratio2)
        + (scale3 * bottom_off[(head - post_pad) * step] * bottom_off[(head - post_pad) * step] * accum_ratio3);
      ++head;
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* scale_data;
  const Dtype* top_data;
  const Dtype* top_diff;
  const Dtype* top_ddiff;
  const Dtype* bottom_data;
  Dtype* bottom_ddiff;

  int n_threads = num_ * height_ * width_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
      scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
      size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff());
  if (Caffe::derivative_compute()) {
    scale_data = scale_.gpu_data();
    top_data = top[0]->gpu_data();
    top_diff = top[0]->gpu_diff();
    top_ddiff = top[0]->gpu_ddiff();
    bottom_data = bottom[0]->gpu_data();
    bottom_ddiff = bottom[0]->mutable_gpu_ddiff();
    Dtype scale1 = Dtype(4. * (beta_ + 1) * beta_ * alpha_ * alpha_/ (size_ * size_));
    Dtype scale2 = Dtype(2. * alpha_ * beta_ / size_);
    Dtype scale3 = (Dtype) scale2 * scale2;

    LRNComputeDdiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
        n_threads, bottom_data, top_data,
        scale_data, top_diff, top_ddiff, num_, channels_, height_, width_,
        size_, -beta_, scale1, scale2, scale3,
        bottom_ddiff);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward_gpu(
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
      Dtype* helper_data_ = this->helper_.mutable_gpu_data();
      Dtype* helper_data2_ = this->helper_.mutable_gpu_diff();
      Dtype* helper_data3_ = this->helper_.mutable_gpu_ddiff();
      
      caffe_gpu_axpy(count, this->alpha_, pool_top_vec_[0]->gpu_data(), helper_data_);
      caffe_gpu_add_scalar(count, this->k_, helper_data_); // nijk
      caffe_gpu_div(count, top[0]->gpu_data(), helper_data_, helper_data2_); // yijk / nijk 
      caffe_gpu_mul(count, top[0]->gpu_diff(), helper_data2_, helper_data2_); // yijk / nijk dE/dyijk
      caffe_gpu_scal(count, (Dtype) -4 * this->beta_ * this->alpha_ / this->size_, helper_data2_); //-4 * beta * alpha / n *  yijk / nijk dE/dyijk
      caffe_gpu_add(count, helper_data2_, bottom[0]->gpu_ddiff(), bottom[0]->mutable_gpu_ddiff());

      caffe_gpu_div(count, top[0]->gpu_data(), helper_data_, helper_data2_); // yijk / nijk 
      caffe_gpu_mul(count, top[0]->gpu_data(), helper_data2_, helper_data2_); // yijk**2 / nijk
      caffe_gpu_mul(count, top[0]->gpu_ddiff(), helper_data2_, helper_data2_); // yijk**2 / nijk d2E/dy2ijk
      caffe_gpu_scal(count, (Dtype) -4 * this->beta_ * this->alpha_ / this->size_, helper_data2_); //-4 * beta * alpha / n *  yijk**2 / nijk d2E/dy2ijk
      caffe_gpu_add(count, helper_data2_, bottom[0]->gpu_ddiff(), bottom[0]->mutable_gpu_ddiff());
    }
  }
}
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);

template void LRNLayer<float>::WithinChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::WithinChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);


INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
