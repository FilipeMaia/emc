#include "emc.h"

__device__ void invert_rot(const real * rot, const real * inv_rot){
  inv_rot[0] = rot[0];
  for(int i = 0;i<3;i++){
    inv_rot[i] = -rot[i];
  }
}


__device__ void cuda_insert_slice_gather(real *model, real *weight, real *slice,
					 int * mask, real w, real *rot, real *x_coordinates,
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
      if (mask[y*x_max + x] == 1) {
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
	if (round_x >= 0 && round_x < model_x &&
	    round_y >= 0 && round_y < model_y &&
	    round_z >= 0 && round_z < model_z) {
	  /* this can cause problems due to non atomic operations */
	  //	  	  model[(round_z*model_x*model_y + round_y*model_x + round_x)] += w * slice[y*x_max + x];	    
		  atomicAdd(&model[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w * slice[y*x_max + x]);
		  //	  	  weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;
		  	  atomicAdd(&weight[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w);
	}
      }//endif
    }
  }
}

__device__ void cuda_insert_slice(real *model, real *weight, real *slice,
				  int * mask, real w, real *rot, real *x_coordinates,
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
      if (mask[y*x_max + x] == 1) {
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
	if (round_x >= 0 && round_x < model_x &&
	    round_y >= 0 && round_y < model_y &&
	    round_z >= 0 && round_z < model_z) {
	  /* this can cause problems due to non atomic operations */
	  //	  	  model[(round_z*model_x*model_y + round_y*model_x + round_x)] += w * slice[y*x_max + x];	    
		  atomicAdd(&model[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w * slice[y*x_max + x]);
		  //	  	  weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;
		  	  atomicAdd(&weight[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w);
	}
      }//endif
    }
  }
}


__global__ void update_slices_kernel(real * images, real * slices, int * mask, real * respons,
				     real * scaling, int N_images, int N_slices, int N_2d,
				     real * slices_total_respons, real * rot,
				     real * x_coord, real * y_coord, real * z_coord,
				     real * model, real * weight,
				     int slice_rows, int slice_cols,
				     int model_x, int model_y, int model_z, real * weights){
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
    __syncthreads();
    cuda_insert_slice(model,weight,&slices[i_slice*N_2d],mask,weights[i_slice]*total_respons,
		      &rot[4*i_slice],x_coord,y_coord,z_coord,
		      slice_rows,slice_cols,model_x,model_y,model_z,tid,step);
  }
  
}



__global__ void update_slices_gather_kernel(real * images, real * slices, int * mask, real * respons,
					    real * scaling, int N_images, int N_slices, int N_2d,
					    real * slices_total_respons, real * rot,
					    real * x_coord, real * y_coord, real * z_coord,
					    real * model, real * weight,
					    int slice_rows, int slice_cols,
					    int model_x, int model_y, int model_z, real * weights){
  /* each block takes care of 1 pixel */
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
    __syncthreads();
    cuda_insert_slice(model,weight,&slices[i_slice*N_2d],mask,weights[i_slice]*total_respons,
		      &rot[4*i_slice],x_coord,y_coord,z_coord,
		      slice_rows,slice_cols,model_x,model_y,model_z,tid,step);
  }
  
}
