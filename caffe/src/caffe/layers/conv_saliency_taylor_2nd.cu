#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void taylor_2nd_kernel_gpu(const int N, const int num, const Dtype * data, const Dtype * diff, const Dtype * ddiff, Dtype * taylor_2nd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    taylor_2nd[i] = ( 0.5 * data[i] * data[i] * ddiff[i] ) - (num * data[i] * diff[i]);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_gpu(const Dtype *  act_data, const Dtype * act_diff, const Dtype *  act_ddiff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_out) {
  Dtype * output_saliency_data = NULL;
  output_saliency_data = output_saliencies_points_.mutable_gpu_data();
  taylor_2nd_kernel_gpu<Dtype><<<CAFFE_GET_BLOCKS(output_saliencies_points_.count()), CAFFE_CUDA_NUM_THREADS>>>(
      output_saliencies_points_.count(), this->num_, act_data, act_diff, act_ddiff, output_saliency_data);

  compute_norm_and_batch_avg_gpu(output_saliency_data, saliency_norm_, taylor_2nd_out);

}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_taylor_2nd_weights_gpu(Blob<Dtype> * weights_n, Blob<Dtype> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, Dtype * taylor_2nd_out) {
  const Dtype* weights = this->blobs_[0]->gpu_data();
  const Dtype* weights_n_diff = weights_n->gpu_diff();
  const Dtype* weights_n_ddiff = weights_n->gpu_ddiff();
  Dtype* points_saliency_data = weights_n->mutable_gpu_data();

  const Dtype* bias;
  const Dtype* bias_n_diff;
  const Dtype* bias_n_ddiff;
  Dtype* bias_saliency_data;

  if (this->mask_term_) {
    weights = weights_masked_.gpu_data();
  }

  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, weights_n_ddiff + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //a * d2E/da2
  }

  caffe_gpu_scal(weights_n->count(), 1/(Dtype)(2*(this->num_)), points_saliency_data);  //1/2N * (a * d2E/da2)
  caffe_gpu_sub(weights_n->count(), points_saliency_data, weights_n_diff, points_saliency_data); //(a/2N * d2E/da2) - 1/N * dE/da
  for (int n = 0; n<this->num_; n++) {
    caffe_gpu_mul(this->blobs_[0]->count(), weights, points_saliency_data + n * this->blobs_[0]->count(), points_saliency_data + n * this->blobs_[0]->count()); //(a**2/2N * d2E/da2) - a/N*dE/da
  }
  caffe_gpu_scal(weights_n->count(), (Dtype)(this->num_), points_saliency_data);

  if (this->saliency_bias_ && this->bias_term_) {
    bias = this->blobs_[1]->gpu_data();
    bias_n_diff = bias_n->gpu_diff();
    bias_n_ddiff = bias_n->gpu_ddiff();
    bias_saliency_data = bias_n->mutable_gpu_data();
    if (this->mask_term_) {
      bias = bias_masked_.mutable_gpu_data();
    }
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_n_ddiff + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), 1/(Dtype)(2*(this->num_)), bias_saliency_data);
    caffe_gpu_sub(bias_n->count(), bias_saliency_data, bias_n_diff, bias_saliency_data);
    for (int n = 0; n<this->num_; n++) {
      caffe_gpu_mul(this->blobs_[1]->count(), bias, bias_saliency_data + n * this->blobs_[1]->count(), bias_saliency_data + n * this->blobs_[1]->count());
    }
    caffe_gpu_scal(bias_n->count(), (Dtype)(this->num_), bias_saliency_data);
  }

  compute_norm_and_batch_avg_weights_gpu(points_saliency_data, bias_saliency_data, saliency_norm_, taylor_2nd_out);
}

template void ConvolutionLayer<float>::compute_taylor_2nd_gpu(const float *  act_data, const float * act_diff, const float *  act_ddiff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * taylor_2nd_out);
template void ConvolutionLayer<double>::compute_taylor_2nd_gpu(const double *  act_data, const double * act_diff, const double *  act_ddiff, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * taylor_2nd_out);

template void ConvolutionLayer<float>::compute_taylor_2nd_weights_gpu(Blob<float> * weights_n, Blob<float> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, float * taylor_2nd_out);
template void ConvolutionLayer<double>::compute_taylor_2nd_weights_gpu(Blob<double> * weights_n, Blob<double> * bias_n, caffe::ConvolutionSaliencyParameter::NORM saliency_norm_, double * taylor_2nd_out);
}  // namespace caffe
