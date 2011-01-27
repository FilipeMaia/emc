#include "emc.h"


__global__ void update_slices_kernel(float * images, float * slices, int * mask, float * respons,
			  float * scaling, int N_images, int N_slices, int N_2d,
			  float * slices_total_respons){
  /* each block takes care of 1 slice */
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int step = blockDim.x;
  float total_respons = 0.0f;
  int i_slice = bid;
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = tid; i < N_2d; i+=step) {
      if (mask[i] != 0) {
	slices[i_slice*N_2d+i] += images[i_image*N_2d+i]*
	  respons[i_slice*N_images+i_image]/scaling[i_image];
      }
    }
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

float cuda_update_slices(sp_matrix ** images, sp_matrix ** slices, sp_imatrix * mask,
			float * respons, float * scaling, int N_images, int N_slices, int N_2d,
			sp_3matrix * model, sp_matrix *x_coordinates, sp_matrix *y_coordinates,
			sp_matrix *z_coordinates, Quaternion **rotations, float * weights,
			sp_3matrix * weight){
  cudaEvent_t begin;
  cudaEvent_t end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  cudaEventRecord (begin,0);
  float * d_images;
  cudaMalloc(&d_images,sizeof(float)*N_2d*N_images);
  for(int i = 0;i<N_images;i++){
    cudaMemcpy(&(d_images[i*N_2d]),images[i]->data,sizeof(float)*N_2d,cudaMemcpyHostToDevice);
  }
  float * d_slices;
  cudaMalloc(&d_slices,sizeof(float)*N_2d*N_slices);
  cudaMemset(d_slices,0,sizeof(float)*N_2d*N_slices);
  int * d_mask;
  cudaMalloc(&d_mask,sizeof(int)*N_2d);
  cudaMemcpy(d_mask,mask->data,sizeof(int)*N_2d,cudaMemcpyHostToDevice);
  float * d_respons;
  cudaMalloc(&d_respons,sizeof(float)*N_slices*N_images);
  cudaMemcpy(d_respons,respons,sizeof(float)*N_slices*N_images,cudaMemcpyHostToDevice);
  float * d_scaling;
  cudaMalloc(&d_scaling,sizeof(float)*N_images);
  cudaMemcpy(d_scaling,scaling,sizeof(float)*N_images,cudaMemcpyHostToDevice);
  int nblocks = N_slices;
  int nthreads = 256;
  float * d_slices_total_respons;
  cudaMalloc(&d_slices_total_respons,sizeof(float)*N_slices);
  update_slices_kernel<<<nblocks,nthreads>>>(d_images, d_slices, d_mask, d_respons,
					     d_scaling, N_images, N_slices, N_2d,
					     d_slices_total_respons);
  cudaError_t status = cudaGetLastError();
  if(status != cudaSuccess){
    printf("CUDA Error: %s\n",cudaGetErrorString(status));
  }
  float slices_total_respons[N_slices];
  cudaMemcpy(slices_total_respons,d_slices_total_respons,sizeof(float)*N_slices,
	     cudaMemcpyDeviceToHost);
  float overal_respons = 0.0;
  for (int i_slice = 0; i_slice < N_slices; i_slice++) {
    overal_respons += slices_total_respons[i_slice];
    if(slices_total_respons[i_slice] > 1e-10){
      cudaMemcpy(slices[i_slice]->data,&(d_slices[i_slice*N_2d]),sizeof(float)*N_2d,
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
  cudaEventRecord(end,0);
  cudaEventSynchronize (end);
  float ms;
  cudaEventElapsedTime (&ms, begin, end);
  printf("cuda slice update time = %fms\n",ms);
  return overal_respons;
}
