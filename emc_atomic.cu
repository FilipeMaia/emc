#include "emc.h"


__device__ inline void atomicFloatAdd(float *address, float val)
{
  int i_val = __float_as_int(val);
  int tmp0 = 0;
  int tmp1;

  while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
    {
      tmp0 = tmp1;
      i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}


__device__ void invert_rot(const real * rot, real * inv_rot){
  inv_rot[0] = rot[0];
  for(int i = 0;i<3;i++){
    inv_rot[i] = -rot[i];
  }
}

__device__ void get_pixel_from_voxel(real detector_distance, 
				     real pixel_size, int x_max, int y_max,
				     const real * voxel, real * pixel){
  const real v_x = voxel[0];
  const real v_y = voxel[1];
  const real v_z = voxel[2];
  real p_x = v_x - 0.5 + x_max/2;
  real p_y = v_y - 0.5 + y_max/2;
  real pixel_r = sqrt(v_x*v_x + v_y*v_y);
  real real_r = pixel_r*pixel_size;
  real angle_r = atan2(real_r,detector_distance);
  real fourier_r = sinf(angle_r);
  real fourier_z = (1.0f - cosf(angle_r));
  real calc_z = fourier_z/fourier_r*pixel_r;
  if(fabs(calc_z - v_z) <= 0.5 &&
     p_x >= 0 && p_x <= x_max-1 &&
     p_y >= 0 && p_y <= y_max-1){
    pixel[0] = p_x;
    pixel[1] = p_y;    
  }else{
    pixel[0] = -1;
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
#if __CUDA_ARCH__ >= 300
	  atomicAdd(&model[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w * slice[y*x_max + x]);
	  //	  	  weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;
	  atomicAdd(&weight[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w);
#else
	  atomicFloatAdd(&model[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w * slice[y*x_max + x]);
	  //	  	  weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;
	  atomicFloatAdd(&weight[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w);

#endif
	}
      }//endif
    }
  }
}




__global__ void insert_slices_non_atomic_kernel(real *models, real *model_weights,
						const real *slices,
						const int * mask,const real *rots,
						const real *x_coordinates,
						const real *y_coordinates, 
						const real *z_coordinates, 
						const real * slices_total_respons,
						const real * slice_weights,
						int slice_rows,
						int slice_cols, int model_x,
						int model_y, int model_z)
{
  /*
    each block takes care of 1 slice
    each block has its own model and model_weights 
  */
  int bid = blockIdx.x;
  real * model = &models[bid*model_x*model_y*model_z];
  real * model_weight = &model_weights[bid*model_x*model_y*model_z];
  const real * slice = &slices[bid*slice_cols*slice_rows];
  const real * rot = &rots[4*bid];
  real w = slices_total_respons[bid]*slice_weights[bid];
  int x_step = 4;
  int y_step = 4;
  const int x_max = slice_rows;
  const int y_max = slice_cols;
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  /* we're going to update pixels in a grid to avoid overlaps */
  for(int x_offset = 0;x_offset < x_step;x_offset++){
    for(int y_offset = 0;y_offset < y_step;y_offset++){
      for (int x = x_offset+threadIdx.x*x_step; x < x_max; x+=x_step*blockDim.x) {
	for (int y = y_offset+threadIdx.y*y_step; y < y_max; y+=y_step*blockDim.y) {
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
	      model[(round_z*model_x*model_y + round_y*model_x + round_x)] += w * slice[y*x_max + x];	    
	      //		  atomicAdd(&model[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w * slice[y*x_max + x]);
	      model_weight[(round_z*model_x*model_y + round_y*model_x + round_x)] += w;
	      //		  	  atomicAdd(&weight[(int)(round_z*model_x*model_y + round_y*model_x + round_x)], w);
	    }
	  }
	}
      }
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
  }
  
}

__global__ void insert_slices_kernel(real * images, real * slices, int * mask, real * respons,
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
  int i_slice = bid;
  real total_respons = slices_total_respons[bid];
  if(total_respons > 1e-10f){
    cuda_insert_slice(model,weight,&slices[i_slice*N_2d],mask,weights[i_slice]*total_respons,
		      &rot[4*i_slice],x_coord,y_coord,z_coord,
		      slice_rows,slice_cols,model_x,model_y,model_z,tid,step);
  }  
}




__device__ void cuda_gather_pixel(real *model, real *weight, real *slices,
				  int * mask, real *rots,real * slices_total_respons,
				  real * weights,int slice_rows, 
				  int slice_cols, int model_x, int model_y, int model_z,
				  int N_slices, real detector_distance, real pixel_size)
{
  real v_x = threadIdx.x - model_x/2 + 0.5f;
  real v_y = blockIdx.y - model_y/2 + 0.5f;
  real v_z = blockIdx.x - model_z/2 + 0.5f;
  int v = threadIdx.x + blockIdx.y*blockDim.x + blockIdx.x*blockDim.x*gridDim.y;
  const int x_max = slice_rows;
  const int y_max = slice_cols;
  const int N_2d = x_max * y_max;
  for (int i_slice = 0; i_slice < N_slices; i_slice++){
    if(slices_total_respons[i_slice] <= 1e-10f){
      continue;
    }
    real inv_rot[4] = {rots[i_slice*4+0],-rots[i_slice*4+1],-rots[i_slice*4+2],-rots[i_slice*4+3]};
    real voxel[3];
    voxel[0] =
      (inv_rot[0]*inv_rot[0] + inv_rot[1]*inv_rot[1] -
       inv_rot[2]*inv_rot[2] - inv_rot[3]*inv_rot[3])*v_x+
      (2.0f*inv_rot[1]*inv_rot[2] -
       2.0f*inv_rot[0]*inv_rot[3])*v_y+
      (2.0f*inv_rot[1]*inv_rot[3] +
       2.0f*inv_rot[0]*inv_rot[2])*v_z;
    voxel[1] =
      (2.0f*inv_rot[1]*inv_rot[2] +
       2.0f*inv_rot[0]*inv_rot[3])*v_x+
      (inv_rot[0]*inv_rot[0] - inv_rot[1]*inv_rot[1] +
       inv_rot[2]*inv_rot[2] - inv_rot[3]*inv_rot[3])*v_y+
      (2.0f*inv_rot[2]*inv_rot[3] -
       2.0f*inv_rot[0]*inv_rot[1])*v_z;
    voxel[2] =
      (2.0f*inv_rot[1]*inv_rot[3] -
       2.0f*inv_rot[0]*inv_rot[2])*v_x+
      (2.0f*inv_rot[2]*inv_rot[3] +
       2.0f*inv_rot[0]*inv_rot[1])*v_y+
      (inv_rot[0]*inv_rot[0] - inv_rot[1]*inv_rot[1] -
       inv_rot[2]*inv_rot[2] + inv_rot[3]*inv_rot[3])*v_z;
    real pixel[2];
    get_pixel_from_voxel(detector_distance,
			 pixel_size, x_max, y_max,
			 voxel, pixel);
    int x = lrint(pixel[0]);
    int y = lrint(pixel[1]);
    if(x != -1 && mask[y * x_max + x] == 1){
      real w = weights[i_slice]*slices_total_respons[i_slice];
      model[v] += w * slices[i_slice*N_2d+y*x_max + x];	    
      weight[v] += w;
    }
  }
}

__global__ void update_slices_gather_kernel(real * images, real * slices, int * mask, real * respons,
					    real * scaling, int N_images, int N_slices, int N_2d,
					    real * slices_total_respons, real * rot,
					    real * model, real * weight,
					    int slice_rows, int slice_cols,
					    int model_x, int model_y, int model_z, real * weights,
					    real detector_distance, real pixel_size){
  /* At the moment this is buggy! */
  /* each block takes care of one pixel line */
  cuda_gather_pixel(model, weight, slices,
		    mask,rot, slices_total_respons, weights,
		    slice_rows,
		    slice_cols, model_x, model_y, model_z,
		    N_slices, detector_distance,  pixel_size);
}
