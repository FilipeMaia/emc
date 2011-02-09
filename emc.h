#pragma once 

#include <spimage.h>

#ifdef __cplusplus
extern "C"{
#endif
typedef struct{
  real q[4];
}Quaternion;

typedef struct{
  int side;
  real wavelength;
  real pixel_size;
  int detector_size;
  real detector_distance;
}Setup;

  void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
		    sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
		    sp_matrix *y_coordinates, sp_matrix *z_coordinates);

  real cuda_update_slices(real * d_images, real * slices, int * d_mask,
			  real * respons, real * d_scaling, int N_images, int N_slices, int N_2d,
			  sp_3matrix * model, real * d_model, 
			  real  *x_coordinates, real *y_coordinates,
			  real *z_coordinates, real *d_rotations, real * weights,
			  real * d_weight,Setup setup,sp_matrix ** images);


  void cuda_update_scaling(real * d_images, real * slices, int * d_mask,
			   real * respons, real * d_scaling, int N_images, int N_slices, int N_2d);
  void cuda_calculate_responsabilities(real * slices, real * images, int * d_mask,
				       real sigma, real * d_scaling, real * d_respons, 
				       int N_2d, int N_images, int N_slices, real * respons);

void cuda_get_slices(sp_3matrix * model, real * d_model, real * d_slices, real * d_rot, 
		     real * d_x_coordinates,
		     real * d_y_coordinates, real * d_z_coordinates, int N_slices);

  void cuda_allocate_slices(real ** slices,Setup setup, int N_slices);
  real cuda_model_max(real * model, int model_size);
  void cuda_allocate_model(real ** d_model, sp_3matrix * model);
  void cuda_allocate_mask(int ** d_mask, sp_imatrix * mask);
  void cuda_reset_model(sp_3matrix * model, real * d_model);
  void cuda_normalize_model(sp_3matrix * model, real * d_model, real * d_weight);
  void cuda_allocate_rotations(real ** d_rotations, Quaternion ** rotations, int N_slices);
  void cuda_allocate_images(real ** d_images, sp_matrix ** images, int N_images);
  void cuda_allocate_coords(real ** d_x, real ** d_y, real ** d_z, sp_matrix * x,
			  sp_matrix * y, sp_matrix * z);
  void cuda_allocate_real(real ** x, int n);
  void cuda_allocate_scaling(real ** scaling, int N_images);
#ifdef __cplusplus
  }
#endif
