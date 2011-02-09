#include "emc.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>


__global__ void update_slices_kernel(real * images, real * slices, int * mask, real * respons,
				     real * scaling, int N_images, int N_slices, int N_2d,
				     real * slices_total_respons, real * rot,
				     real * x_coord, real * y_coord, real * z_coord,
				     real * model, real * weight,
				     int slice_rows, int slice_cols,
				     int model_x, int model_y, int model_z, real * weights);

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
				     real * scaling, int N_images, int N_slices, int N_2d,
				     real * slices_total_respons, real * rot,
				     real * x_coord, real * y_coord, real * z_coord,
				     real * model, real * weight,
				     int slice_rows, int slice_cols,
				     int model_x, int model_y, int model_z, real * weights);


template<typename T>
__device__ void inblock_reduce(T * data){
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1){
    if (threadIdx.x < s){
      data[threadIdx.x] += data[threadIdx.x + s];
    }
    __syncthreads();
  }  
}

template<typename T>
__device__ void inblock_maximum(T * data){
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1){
    if (threadIdx.x < s){
      if(data[threadIdx.x] < data[threadIdx.x + s]){
	data[threadIdx.x] = data[threadIdx.x + s];
      }
    }
    __syncthreads();
  }  
}



__device__ void cuda_get_slice(real *model, real *slice,
				  real *rot, real *x_coordinates,
				  real *y_coordinates, real *z_coordinates, int slice_rows,
				  int slice_cols, int model_x, int model_y, int model_z,
				  int tid, int step)
{
  const int x_max = slice_rows;
  const int y_max = slice_cols;
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  for (int x = 0; x < x_max; x++) {
    for (int y = tid; y < y_max; y+=step) {
      /* This is just a matrix multiplication with rot */
      new_x =
	(rot[0]*rot[0] + rot[1]*rot[1] -
	 rot[2]*rot[2] - rot[3]*rot[3])*x_coordinates[y*x_max+x] +
	(2.0f*rot[1]*rot[2] -
	 2.0f*rot[0]*rot[3])*y_coordinates[y*x_max+x] +
	(2.0f*rot[1]*rot[3] +
	 2.0f*rot[0]*rot[2])*z_coordinates[y*x_max+x];
      new_y =
	(2.0f*rot[1]*rot[2] +
	 2.0f*rot[0]*rot[3])*x_coordinates[y*x_max+x] +
	(rot[0]*rot[0] - rot[1]*rot[1] +
	 rot[2]*rot[2] - rot[3]*rot[3])*y_coordinates[y*x_max+x] +
	(2.0f*rot[2]*rot[3] -
	 2.0f*rot[0]*rot[1])*z_coordinates[y*x_max+x];
      new_z =
	(2.0f*rot[1]*rot[3] -
	 2.0f*rot[0]*rot[2])*x_coordinates[y*x_max+x] +
	(2.0f*rot[2]*rot[3] +
	 2.0f*rot[0]*rot[1])*y_coordinates[y*x_max+x] +
	(rot[0]*rot[0] - rot[1]*rot[1] -
	 rot[2]*rot[2] + rot[3]*rot[3])*z_coordinates[y*x_max+x];
      round_x = roundf(model_x/2.0f + 0.5f + new_x);
      round_y = roundf(model_y/2.0f + 0.5f + new_y);
      round_z = roundf(model_z/2.0f + 0.5f + new_z);
      if (round_x > 0 && round_x < model_x &&
	  round_y > 0 && round_y < model_y &&
	  round_z > 0 && round_z < model_z) {
	slice[y*x_max+x] = model[(round_z*model_x*model_y + round_y*model_x + round_x)];
      }else{
	slice[y*x_max+x] = 0.0f;
      }
    }
  }
}

__global__ void get_slices_kernel(real * model, real * slices, real *rot, real *x_coordinates,
				  real *y_coordinates, real *z_coordinates, int slice_rows,
				  int slice_cols, int model_x, int model_y, int model_z){
  int bid = blockIdx.x;
  int i_slice = bid;
  int tid = threadIdx.x;
  int step = blockDim.x;
  int N_2d = slice_rows*slice_cols;
  cuda_get_slice(model,&slices[N_2d*i_slice],&rot[4*i_slice],x_coordinates,
		 y_coordinates,z_coordinates,slice_rows,slice_cols,model_x,model_y,
		 model_z,tid,step);
}

/* This responsability does not yet take scaling of patterns into accoutnt. */
__device__ void cuda_calculate_responsability_absolute(float *slice, float *image, int *mask, real sigma, real scaling, int N_2d, int tid, int step, real * sum_cache, int * count_cache)
{
  real sum = 0.0;
  const int i_max = N_2d;
  int count = 0;
  for (int i = tid; i < i_max; i+=step) {
    if (mask[i] != 0) {
      sum += pow(slice[i] - image[i]/scaling,2);
      count++;
    }
  }
  sum_cache[tid] = sum;
  count_cache[tid] = count;
  //  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

__global__ void calculate_responsabilities_kernel(float * slices, float * images, int * mask,
						  real sigma, real * scaling, real * respons, 
						  int N_2d){
  __shared__ real sum_cache[256];
  __shared__ int count_cache[256];
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = blockIdx.x;
  int i_slice = blockIdx.y;
  int N_images = gridDim.x;
  cuda_calculate_responsability_absolute(&slices[i_slice*N_2d],
					 &images[i_image*N_2d],mask,
					 sigma,scaling[i_image], N_2d, tid,step,
					 sum_cache,count_cache);
  inblock_reduce(sum_cache);
  inblock_reduce(count_cache);
  
  if(tid == 0){
    respons[i_slice*N_images+i_image] = -sum_cache[0]/2.0/(real)count_cache[0]/pow(sigma,2);
  }   
}



void cuda_calculate_responsabilities(real * d_slices, real * d_images, int * d_mask,
				     real sigma, real * d_scaling, real * d_respons, 
				     int N_2d, int N_images, int N_slices, real * respons){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  dim3 nblocks(N_images,N_slices);
  int nthreads = 256;
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);
  calculate_responsabilities_kernel<<<nblocks,nthreads>>>(d_slices,d_images,d_mask,
							  sigma,d_scaling,d_respons,
							  N_2d);
  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  printf("cuda kernel calc respons time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error: %s\n",cudaGetErrorString(status));
  }
  cudaMemcpy(respons,d_respons,sizeof(real)*N_slices*N_images,cudaMemcpyDeviceToHost);
  real respons_sum = 0;
  for(int i = 0;i<N_slices*N_images;i++){
    respons_sum += respons[i];
  }
  printf("respons_sum = %f\n",respons_sum);
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  real ms;
  cudaEventElapsedTime (&ms, begin, end);
  printf("cuda calc respons time = %fms\n",ms);
}  

__global__ void slice_weighting_kernel(real * images, real * slices,int * mask,
		     real * respons, real * scaling,
				       int N_slices, int N_2d, int N_images){
  __shared__ real image_power[256];
  __shared__ real correlation[256];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  int i_image = bid;  
  real weighted_power = 0;

  image_power[tid] = 0.0;
  for (int i = tid; i < N_2d; i+=step) {
    if (mask[i] != 0) {
      image_power[tid] += pow(images[i_image*N_2d+i],2);
    }
  }
  inblock_reduce(image_power);
  for (int i_slice = 0; i_slice < N_slices; i_slice++) { 
    correlation[tid] = 0.0;
    for (int i = tid; i < N_2d; i+=step) {
      if (mask[i] != 0) {
	correlation[tid] += images[i_image*N_2d+i]*slices[i_slice*N_2d+i];
      }
    }
    inblock_reduce(correlation);
    if(tid == 0){
      weighted_power += respons[i_slice*N_images+i_image]*correlation[tid];
    }
  }  
  if(tid == 0){
    scaling[i_image] = image_power[tid]/weighted_power;
  }
}

void cuda_update_scaling(real * d_images, real * d_slices, int * d_mask,
			 real * d_respons, real * d_scaling, int N_images, int N_slices, int N_2d,
			 real * scaling){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  int nblocks = N_images;
  int nthreads = 256;
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);
  slice_weighting_kernel<<<nblocks,nthreads>>>(d_images,d_slices,d_mask,
			 d_respons, d_scaling,
			 N_slices,N_2d, N_images);
  cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,cudaMemcpyDeviceToHost);
  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  printf("cuda kernel update scaling time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error: %s\n",cudaGetErrorString(status));
  }
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  real ms;
  cudaEventElapsedTime (&ms, begin, end);
  printf("cuda update scaling time = %fms\n",ms);
}


void cuda_get_slices(sp_3matrix * model, real * d_model, real * d_slices, real * d_rot, 
		     real * d_x_coordinates,
		     real * d_y_coordinates, real * d_z_coordinates, int N_slices){
  
  int rows = sp_3matrix_x(model);
  int cols = sp_3matrix_y(model);
  int N_2d = sp_3matrix_x(model)*sp_3matrix_y(model);
  int nblocks = N_slices;
  int nthreads = 256;
  get_slices_kernel<<<nblocks,nthreads>>>(d_model, d_slices, d_rot,d_x_coordinates,
					  d_y_coordinates,d_z_coordinates,
					  rows,cols,
					  sp_3matrix_x(model),sp_3matrix_y(model),
					  sp_3matrix_z(model));
}

real cuda_update_slices(real * d_images, real * d_slices, int * d_mask,
			real * d_respons, real * d_scaling, int N_images, int N_slices, int N_2d,
			sp_3matrix * model, real * d_model,
			real *d_x_coordinates, real *d_y_coordinates,
			real *d_z_coordinates, real *d_rot, real * weights,
			real * d_weight, Setup setup, sp_matrix ** images){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  dim3 nblocks = N_slices;
  int nthreads = 256;
  real * d_slices_total_respons;
  cudaMalloc(&d_slices_total_respons,sizeof(real)*N_slices);

  real * d_weights;
  cudaMalloc(&d_weights,sizeof(real)*N_slices);
  cudaMemcpy(d_weights,weights,sizeof(real)*N_slices,cudaMemcpyHostToDevice);



  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  update_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
					     d_scaling, N_images, N_slices, N_2d,
					     d_slices_total_respons, d_rot,d_x_coordinates,
					     d_y_coordinates,d_z_coordinates,d_model, d_weight,
					     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
					     sp_3matrix_x(model),sp_3matrix_y(model),
					     sp_3matrix_z(model),d_weights);  
  cudaThreadSynchronize();
  insert_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
					     d_scaling, N_images, N_slices, N_2d,
					     d_slices_total_respons, d_rot,d_x_coordinates,
					     d_y_coordinates,d_z_coordinates,d_model, d_weight,
					     sp_matrix_rows(images[0]),sp_matrix_cols(images[0]),
					     sp_3matrix_x(model),sp_3matrix_y(model),
					     sp_3matrix_z(model),d_weights);  
  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  printf("cuda kernel slice update time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error: %s\n",cudaGetErrorString(status));
  }
  real slices_total_respons[N_slices];
  cudaMemcpy(slices_total_respons,d_slices_total_respons,sizeof(real)*N_slices,
	     cudaMemcpyDeviceToHost);
  real overal_respons = 0.0;
  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
    overal_respons += slices_total_respons[i_slice];
  }
  cudaMemcpy(model->data,d_model,sizeof(real)*sp_3matrix_size(model),cudaMemcpyDeviceToHost);

  cudaFree(d_slices_total_respons);
  cudaFree(d_weights);
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  real ms;
  cudaEventElapsedTime (&ms, begin, end);
  printf("cuda slice update time = %fms\n",ms);
  return overal_respons;
}

real cuda_model_max(real * model, int model_size){
  thrust::device_ptr<real> p(model);
  real max = thrust::reduce(p, p+model_size, real(0), thrust::maximum<real>());
  return max;
}

void cuda_allocate_slices(real ** slices,Setup setup,int N_slices){
  cudaMalloc(slices,sizeof(real)*setup.side*setup.side*N_slices);  
}

void cuda_allocate_model(real ** d_model, sp_3matrix * model){
  cudaMalloc(d_model,sizeof(real)*sp_3matrix_size(model));
  cudaMemcpy(*d_model,model->data,sizeof(real)*sp_3matrix_size(model),cudaMemcpyHostToDevice);
}

void cuda_allocate_mask(int ** d_mask, sp_imatrix * mask){
  cudaMalloc(d_mask,sizeof(int)*sp_imatrix_size(mask));
  cudaMemcpy(*d_mask,mask->data,sizeof(int)*sp_imatrix_size(mask),cudaMemcpyHostToDevice);
}

void cuda_allocate_rotations(real ** d_rotations, Quaternion ** rotations,  int N_slices){

  cudaMalloc(d_rotations,sizeof(real)*4*N_slices);
  for(int i = 0;i<N_slices;i++){
    cudaMemcpy(&(*d_rotations)[4*i],rotations[i]->q,sizeof(real)*4,cudaMemcpyHostToDevice);
  }
}

void cuda_allocate_images(real ** d_images, sp_matrix ** images,  int N_images){

  cudaMalloc(d_images,sizeof(real)*sp_matrix_size(images[0])*N_images);
  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(*d_images)[sp_matrix_size(images[0])*i],images[i]->data,sizeof(real)*sp_matrix_size(images[0]),cudaMemcpyHostToDevice);
  }
}

void cuda_allocate_coords(real ** d_x, real ** d_y, real ** d_z, sp_matrix * x,
			  sp_matrix * y, sp_matrix * z){
  cudaMalloc(d_x,sizeof(real)*sp_matrix_size(x));
  cudaMalloc(d_y,sizeof(real)*sp_matrix_size(x));
  cudaMalloc(d_z,sizeof(real)*sp_matrix_size(x));
  cudaMemcpy(*d_x,x->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
  cudaMemcpy(*d_y,y->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
  cudaMemcpy(*d_z,z->data,sizeof(real)*sp_matrix_size(x),cudaMemcpyHostToDevice);
}

void cuda_reset_model(sp_3matrix * model, real * d_model){
  cudaMemset(d_model,0,sizeof(real)*sp_3matrix_size(model));
}

__global__ void cuda_normalize_model_kernel(real * model, real * weight, int n){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(weight[i] > 0.0f){
    model[i] /= weight[i];
  }else{
    model[i] = 0.0f;
  }
}
void cuda_normalize_model(sp_3matrix * model, real * d_model, real * d_weight){
  int n = sp_3matrix_size(model);
  int nthreads = 256;
  int nblocks = (n+nthreads-1)/nthreads;
  cuda_normalize_model_kernel<<<nblocks,nthreads>>>(d_model,d_weight,n);
  cudaThreadSynchronize();
  thrust::device_ptr<real> p(d_model);
  real model_sum = thrust::reduce(p, p+n, real(0), thrust::plus<real>());
  model_sum /= n;
  /* model /= model_sum; */
  thrust::transform(p, p+n,thrust::make_constant_iterator(1.0f/model_sum), p, thrust::multiplies<real>()); 
}

void cuda_allocate_real(real ** x, int n){
  cudaMalloc(x,n);
}
			  
void cuda_allocate_scaling(real ** d_scaling, int N_images){
  cudaMalloc(d_scaling,N_images*sizeof(real));
  thrust::device_ptr<real> p(*d_scaling);
  thrust::fill(p, p+N_images, real(1));
}

__global__ void cuda_normalize_responsabilities_kernel(real * respons, int N_slices, int N_images){
  __shared__ real cache[256];
  int i_image = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  cache[tid] = -1.0e10f;
  for(int i_slice = tid;i_slice < N_slices;i_slice += step){
    if(cache[tid] < respons[i_slice*N_images+i_image]){
      cache[tid] = respons[i_slice*N_images+i_image];
    }
  }
  inblock_maximum(cache);
  real max_resp = cache[0];
  for (int i_slice = tid; i_slice < N_slices; i_slice+= step) {
    respons[i_slice*N_images+i_image] -= max_resp;
  }
  cache[tid] = 0;
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    if (respons[i_slice*N_images+i_image] > -1.0e10f) {
      respons[i_slice*N_images+i_image] = expf(respons[i_slice*N_images+i_image]);
      cache[tid] += respons[i_slice*N_images+i_image];
    } else {
      respons[i_slice*N_images+i_image] = 0.0f;
    }
  }
  inblock_reduce(cache);
  real sum = cache[0];
  for (int i_slice = tid; i_slice < N_slices; i_slice+=step) {
    respons[i_slice*N_images+i_image] /= sum;
  }
  
}

void cuda_normalize_responsabilities(real * d_respons, int N_slices, int N_images){
  int nblocks = N_images;
  int nthreads = 256;
  cuda_normalize_responsabilities_kernel<<<nblocks,nthreads>>>(d_respons,N_slices,N_images);
}

// x_log_x<T> computes the f(x) -> x*log(x)
template <typename T>
struct x_log_x
{
  __host__ __device__
  T operator()(const T& x) const { 
    if(x > 0){
      return x * logf(x);
    }else{
      return 0;
    }
  }
};

real cuda_total_respons(real * d_respons, real * respons,int n){
  thrust::device_ptr<real> p(d_respons);
  x_log_x<real> unary_op;
  thrust::plus<real> binary_op;
  real init = 0;
  // Calculates sum_0^n d_respons*log(d_respons)
  return thrust::transform_reduce(p, p+n, unary_op, init, binary_op);
}