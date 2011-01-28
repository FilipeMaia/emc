#include "emc.h"


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
  __syncthreads();
  for(unsigned int s=blockDim.x/2; s>0; s>>=1){
    if (tid < s){
      image_power[tid] += image_power[tid + s];
    }
    __syncthreads();
  }  
  for (int i_slice = 0; i_slice < N_slices; i_slice++) { 
    correlation[tid] = 0.0;
    for (int i = tid; i < N_2d; i+=step) {
      if (mask[i] != 0) {
	correlation[tid] += images[i_image*N_2d+i]*slices[i_slice*N_2d+i];
      }
    }
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
      if (tid < s){
	correlation[tid] += correlation[tid + s];
      }
      __syncthreads();
    }  
    if(tid == 0){
      weighted_power += respons[i_slice*N_images+i_image]*correlation[tid];
    }
  }  
  if(tid == 0){
    scaling[i_image] = image_power[tid]/weighted_power;
  }
}

void cuda_update_scaling(sp_matrix ** images, sp_matrix ** slices, sp_imatrix * mask,
			 real * respons, real * scaling, int N_images, int N_slices, int N_2d){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  real * d_images;
  cudaMalloc(&d_images,sizeof(real)*N_2d*N_images);
  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(d_images[i*N_2d]),images[i]->data,sizeof(real)*N_2d,cudaMemcpyHostToDevice);
  }
  real * d_slices;
  cudaMalloc(&d_slices,sizeof(real)*N_2d*N_slices);
  cudaMemset(d_slices,0,sizeof(real)*N_2d*N_slices);
  int * d_mask;
  cudaMalloc(&d_mask,sizeof(int)*N_2d);
  cudaMemcpy(d_mask,mask->data,sizeof(int)*N_2d,cudaMemcpyHostToDevice);
  real * d_respons;
  cudaMalloc(&d_respons,sizeof(real)*N_slices*N_images);
  cudaMemcpy(d_respons,respons,sizeof(real)*N_slices*N_images,cudaMemcpyHostToDevice);
  real * d_scaling;
  cudaMalloc(&d_scaling,sizeof(real)*N_images);
  cudaMemcpy(d_scaling,scaling,sizeof(real)*N_images,cudaMemcpyHostToDevice);
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
  cudaEventRecord(k_end,0);
  cudaEventSynchronize(k_end);
  real k_ms;
  cudaEventElapsedTime (&k_ms, k_begin, k_end);
  printf("cuda kernel slice update time = %fms\n",k_ms);

  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error: %s\n",cudaGetErrorString(status));
  }
  cudaMemcpy(scaling,d_scaling,sizeof(real)*N_images,
	     cudaMemcpyDeviceToHost);
  cudaFree(d_images);
  cudaFree(d_slices);
  cudaFree(d_mask);
  cudaFree(d_respons);
  cudaFree(d_scaling);
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  real ms;
  cudaEventElapsedTime (&ms, begin, end);
  printf("cuda slice update time = %fms\n",ms);
}

__global__ void update_slices_kernel(real * images, real * slices, int * mask, real * respons,
			  real * scaling, int N_images, int N_slices, int N_2d,
			  real * slices_total_respons){
  /* each block takes care of 1 slice */
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  real total_respons = 0.0f;
  int i_slice = bid;
  for (int i = tid; i < N_2d; i+=step) {
    if (mask[i] != 0) {
      real sum = 0;
      for (int i_image = 0; i_image < N_images; i_image++) {
	sum += images[i_image*N_2d+i]*
	  respons[i_slice*N_images+i_image]/scaling[i_image];
      }
      slices[i_slice*N_2d+i] = sum;
    }
  }
  for (int i_image = 0; i_image < N_images; i_image++) {
    total_respons += respons[i_slice*N_images+i_image];
  }
  if(tid == 0){    
    slices_total_respons[bid] =  total_respons;
  }  
  if(total_respons > 1e-10f){
    for (int i = tid; i < N_2d; i+=step) {
      if (mask[i] != 0) {
	slices[i_slice*N_2d+i] /= total_respons;
      }
    }
  }
}

real cuda_update_slices(sp_matrix ** images, sp_matrix ** slices, sp_imatrix * mask,
			real * respons, real * scaling, int N_images, int N_slices, int N_2d,
			sp_3matrix * model, sp_matrix *x_coordinates, sp_matrix *y_coordinates,
			sp_matrix *z_coordinates, Quaternion **rotations, real * weights,
			sp_3matrix * weight){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  real * d_images;
  cudaMalloc(&d_images,sizeof(real)*N_2d*N_images);
  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(d_images[i*N_2d]),images[i]->data,sizeof(real)*N_2d,cudaMemcpyHostToDevice);
  }
  real * d_slices;
  cudaMalloc(&d_slices,sizeof(real)*N_2d*N_slices);
  cudaMemset(d_slices,0,sizeof(real)*N_2d*N_slices);
  int * d_mask;
  cudaMalloc(&d_mask,sizeof(int)*N_2d);
  cudaMemcpy(d_mask,mask->data,sizeof(int)*N_2d,cudaMemcpyHostToDevice);
  real * d_respons;
  cudaMalloc(&d_respons,sizeof(real)*N_slices*N_images);
  cudaMemcpy(d_respons,respons,sizeof(real)*N_slices*N_images,cudaMemcpyHostToDevice);
  real * d_scaling;
  cudaMalloc(&d_scaling,sizeof(real)*N_images);
  cudaMemcpy(d_scaling,scaling,sizeof(real)*N_images,cudaMemcpyHostToDevice);
  int nblocks = N_slices;
  int nthreads = 256;
  real * d_slices_total_respons;
  cudaMalloc(&d_slices_total_respons,sizeof(real)*N_slices);
  cudaEvent_t k_begin;
  cudaEvent_t k_end;
  cudaEventCreate(&k_begin);
  cudaEventCreate(&k_end);
  cudaEventRecord (k_begin,0);

  update_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
					     d_scaling, N_images, N_slices, N_2d,
					     d_slices_total_respons);
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
    if(slices_total_respons[i_slice] > 1e-10){
      cudaMemcpy(slices[i_slice]->data,&(d_slices[i_slice*N_2d]),sizeof(real)*N_2d,
		 cudaMemcpyDeviceToHost);
      cudaError_t status = cudaGetLastError();
      if(status != cudaSuccess){
	printf("CUDA Error: %s\n",cudaGetErrorString(status));
      }
      insert_slice(model, weight, slices[i_slice], mask, 
		   weights[i_slice]*slices_total_respons[i_slice],
		   rotations[i_slice], x_coordinates, y_coordinates, z_coordinates);
    }
  }
  cudaFree(d_images);
  cudaFree(d_slices);
  cudaFree(d_mask);
  cudaFree(d_respons);
  cudaFree(d_scaling);
  cudaFree(d_slices_total_respons);
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  real ms;
  cudaEventElapsedTime (&ms, begin, end);
  printf("cuda slice update time = %fms\n",ms);
  return overal_respons;
}
