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

  real cuda_update_slices(sp_matrix ** images, real * slices, sp_imatrix * mask,
			  real * respons, real * scaling, int N_images, int N_slices, int N_2d,
			  sp_3matrix * model, sp_matrix *x_coordinates, sp_matrix *y_coordinates,
			  sp_matrix *z_coordinates, Quaternion **rotations, real * weights,
			  sp_3matrix * weight);
  void cuda_update_scaling(sp_matrix ** images, real * slices, sp_imatrix * mask,
			   float * respons, float * scaling, int N_images, int N_slices, int N_2d);
  void cuda_calculate_responsabilities(real * slices, sp_matrix ** images, sp_imatrix * mask,
				       real sigma, real * scaling, real * respons, 
				       int N_2d, int N_images, int N_slices);

  void cuda_get_slices(sp_3matrix * model, real * slices, Quaternion ** rot, 
		       sp_matrix * x_coordinates,
		       sp_matrix * y_coordinates, sp_matrix * z_coordinates, int N_slices);

#ifdef __cplusplus
  }
#endif
