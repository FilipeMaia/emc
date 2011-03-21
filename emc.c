//#include "fragmentation.h"
#include <spimage.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include "emc.h"
//#include "rotations.h"
#include <libconfig.h>

void calculate_coordinates(int side, real pixel_size, real detector_distance, real wavelength,
			   sp_matrix *x_coordinates,
			   sp_matrix *y_coordinates, sp_matrix *z_coordinates) {
  const int x_max = side;
  const int y_max = side;
  real pixel_r, real_r, fourier_r, angle_r, fourier_z;
  real pixel_x, pixel_y, pixel_z;
  //tabulate angle later
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      pixel_r = sqrt(pow((real)(x-x_max/2)+0.5,2) + pow((real)(y-y_max/2)+0.5,2));
      real_r = pixel_r*pixel_size;
      angle_r = atan2(real_r,detector_distance);
      fourier_r = sin(angle_r)/wavelength;
      fourier_z = (1. - cos(angle_r))/wavelength;

      pixel_x = (real)(x-x_max/2)+0.5;
      pixel_y = (real)(y-y_max/2)+0.5;
      pixel_z = fourier_z/fourier_r*pixel_r;
      sp_matrix_set(x_coordinates,x,y,pixel_x);
      sp_matrix_set(y_coordinates,x,y,pixel_y);
      sp_matrix_set(z_coordinates,x,y,pixel_z);
    }
  }
}


void get_pixel_from_voxel(real detector_distance, real wavelength,
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
  real fourier_r = sinf(angle_r)/wavelength;
  real fourier_z = (1.0f - cosf(angle_r))/wavelength;
  real calc_z = fourier_z/fourier_r*pixel_r;
  if(fabs(calc_z - v_z) <= 0.5){
    pixel[0] = p_x;
    pixel[1] = p_y;    
  }else{
    pixel[0] = -1;
  }

}

void test_get_pixel_from_voxel(int side, real pixel_size, real detector_distance, real wavelength,
			       sp_matrix *x_coordinates, sp_matrix *y_coordinates, sp_matrix *z_coordinates) {
  const int x_max = side;
  const int y_max = side;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      real voxel[3];
      voxel[0] = rint(sp_matrix_get(x_coordinates,x,y));
      voxel[1] = rint(sp_matrix_get(y_coordinates,x,y));
      voxel[2] = rint(sp_matrix_get(z_coordinates,x,y));
      real pixel[2];
      get_pixel_from_voxel(detector_distance, wavelength,
			   pixel_size, x_max, y_max,
			   voxel, pixel);
      if(lrint(pixel[0]) != x){
	printf("x - %f and %d don't match\n",pixel[0],x);
      }
      if(lrint(pixel[1]) != y){
	printf("y - %f and %d don't match\n",pixel[1],y);
      }      
    }
  }
}

void get_slice(sp_3matrix *model, sp_matrix *slice, Quaternion *rot,
	       sp_matrix *x_coordinates, sp_matrix *y_coordinates,
	       sp_matrix *z_coordinates)
{
  const int x_max = sp_matrix_rows(slice);
  const int y_max = sp_matrix_cols(slice);
  int pixel_r;
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      /* This is just a matrix multiplication with rot */
      new_x =
	(rot->q[0]*rot->q[0] + rot->q[1]*rot->q[1] -
	 rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +
	(2.0*rot->q[1]*rot->q[2] -
	 2.0*rot->q[0]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +
	(2.0*rot->q[1]*rot->q[3] +
	 2.0*rot->q[0]*rot->q[2])*sp_matrix_get(z_coordinates,x,y);
      new_y =
	(2.0*rot->q[1]*rot->q[2] +
	 2.0*rot->q[0]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +
	(rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] +
	 rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +
	(2.0*rot->q[2]*rot->q[3] -
	 2.0*rot->q[0]*rot->q[1])*sp_matrix_get(z_coordinates,x,y);
      new_z =
	(2.0*rot->q[1]*rot->q[3] -
	 2.0*rot->q[0]*rot->q[2])*sp_matrix_get(x_coordinates,x,y) +
	(2.0*rot->q[2]*rot->q[3] +
	 2.0*rot->q[0]*rot->q[1])*sp_matrix_get(y_coordinates,x,y) +
	(rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] -
	 rot->q[2]*rot->q[2] + rot->q[3]*rot->q[3])*sp_matrix_get(z_coordinates,x,y);
      round_x = round((real)sp_3matrix_x(model)/2.0 + 0.5 + new_x);
      round_y = round((real)sp_3matrix_y(model)/2.0 + 0.5 + new_y);
      round_z = round((real)sp_3matrix_z(model)/2.0 + 0.5 + new_z);
      if (round_x > 0 && round_x < sp_3matrix_x(model) &&
	  round_y > 0 && round_y < sp_3matrix_y(model) &&
	  round_z > 0 && round_z < sp_3matrix_z(model)) {
	sp_matrix_set(slice,x,y,sp_3matrix_get(model,round_x,round_y,round_z));
      } else {
	sp_matrix_set(slice,x,y,0.0);
      }
    }
  }
}

void insert_slice(sp_3matrix *model, sp_3matrix *weight, sp_matrix *slice,
		  sp_imatrix * mask, real w, Quaternion *rot, sp_matrix *x_coordinates,
		  sp_matrix *y_coordinates, sp_matrix *z_coordinates)
{
  const int x_max = sp_matrix_rows(slice);
  const int y_max = sp_matrix_cols(slice);
  int pixel_r;
  //tabulate angle later
  real new_x, new_y, new_z;
  int round_x, round_y, round_z;
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      if (sp_imatrix_get(mask,x,y) == 1) {
	/* This is just a matrix multiplication with rot */
	new_x =
	  (rot->q[0]*rot->q[0] + rot->q[1]*rot->q[1] -
	   rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (2.0*rot->q[1]*rot->q[2] -
	   2.0*rot->q[0]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (2.0*rot->q[1]*rot->q[3] +
	   2.0*rot->q[0]*rot->q[2])*sp_matrix_get(z_coordinates,x,y);
	new_y =
	  (2.0*rot->q[1]*rot->q[2] +
	   2.0*rot->q[0]*rot->q[3])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] +
	   rot->q[2]*rot->q[2] - rot->q[3]*rot->q[3])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (2.0*rot->q[2]*rot->q[3] -
	   2.0*rot->q[0]*rot->q[1])*sp_matrix_get(z_coordinates,x,y);
	new_z =
	  (2.0*rot->q[1]*rot->q[3] -
	   2.0*rot->q[0]*rot->q[2])*sp_matrix_get(x_coordinates,x,y) +/*((real)(x-x_max/2)+0.5)+*/
	  (2.0*rot->q[2]*rot->q[3] +
	   2.0*rot->q[0]*rot->q[1])*sp_matrix_get(y_coordinates,x,y) +/*((real)(y-y_max/2)+0.5)+*/
	  (rot->q[0]*rot->q[0] - rot->q[1]*rot->q[1] -
	   rot->q[2]*rot->q[2] + rot->q[3]*rot->q[3])*sp_matrix_get(z_coordinates,x,y);
	round_x = round((real)sp_3matrix_x(model)/2.0 + 0.5 + new_x);
	round_y = round((real)sp_3matrix_y(model)/2.0 + 0.5 + new_y);
	round_z = round((real)sp_3matrix_z(model)/2.0 + 0.5 + new_z);
	if (round_x >= 0 && round_x < sp_3matrix_x(model) &&
	    round_y >= 0 && round_y < sp_3matrix_y(model) &&
	    round_z >= 0 && round_z < sp_3matrix_z(model)) {
	  sp_3matrix_set(model,round_x,round_y,round_z,
			 sp_3matrix_get(model,round_x,round_y,round_z)+w*sp_matrix_get(slice,x,y));
	  sp_3matrix_set(weight,round_x,round_y,round_z,sp_3matrix_get(weight,round_x,round_y,round_z)+w);
	}
      }//endif
    }
  }
}

real update_slice(sp_matrix ** images, sp_matrix ** slices, sp_imatrix * mask,
		  real * respons, real * scaling,
		  int N_images, int N_2d, int i_slice){
  real total_respons = 0.0;
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] != 0) {
	slices[i_slice]->data[i] += images[i_image]->data[i]*
	  respons[i_slice*N_images+i_image]/scaling[i_image];
      }
    }
    total_respons += respons[i_slice*N_images+i_image];
  }
  return total_respons;
}

real slice_weighting(sp_matrix ** images, sp_matrix ** slices,sp_imatrix * mask,
		     real * respons, real * scaling,
		     int N_slices, int N_2d, int i_image, int N_images){
  real weighted_power = 0;
  for (int i_slice = 0; i_slice < N_slices; i_slice++) { 
    real correlation = 0.0;
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] != 0) {
	correlation += images[i_image]->data[i]*slices[i_slice]->data[i];
      }
    }
    weighted_power += respons[i_slice*N_images+i_image]*correlation;
  }  
  return weighted_power;
}

void calculate_normalization(sp_matrix **images, int size, sp_matrix *average, sp_matrix *scale)
{
  /* create radius matrix */
  sp_matrix *radius = sp_matrix_alloc(sp_matrix_rows(images[0]),sp_matrix_cols(images[0]));
  const int x_max = sp_matrix_rows(radius);
  const int y_max = sp_matrix_cols(radius);
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      sp_matrix_set(radius,x,y,sqrt(pow((real)x-(real)x_max/2.0+0.5,2) +
				    pow((real)y-(real)y_max/2.0+0.5,2)));
    }
  }

  /* calculate average */
  const int length = ceil(sqrt(pow(x_max,2)+pow(y_max,2))/2.0);
  real histogram[length];
  int count[length];
  for (int i = 0; i < length; i++) {
    histogram[i] = 0.0;
    count[i] = 0;
  }
  for (int i_image = 0; i_image < size; i_image++) {
    for (int x = 0; x < x_max; x++) {
      for (int y = 0; y < y_max; y++) {
	histogram[(int)floor(sp_matrix_get(radius,x,y))] += sp_matrix_get(images[i_image],x,y);
	count[(int)floor(sp_matrix_get(radius,x,y))] += 1;
      }
    }
  }
  printf("length = %d\n",length);
  for (int i = 0; i < length; i++) {
    if (count[i] > 0) {
      histogram[i] /= count[i];
    }
  }
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      sp_matrix_set(average,x,y,histogram[(int)floor(sp_matrix_get(radius,x,y))]);
    }
  }


  /* calculate scale */
  for (int i = 0; i < length; i++) {
    histogram[i] = 0.0;
    count[i] = 0;
  }
  for (int i_image = 0; i_image < size; i_image++) {
    for (int x = 0; x < x_max; x++) {
      for (int y = 0; y < y_max; y++) {
	histogram[(int)floor(sp_matrix_get(radius,x,y))] += pow(sp_matrix_get(images[i_image],x,y) -
							   sp_matrix_get(average,x,y),2);
	count[(int)floor(sp_matrix_get(radius,x,y))] += 1;
      }
    }
  }
  for (int i = 0; i < length; i++) {
    histogram[i] /= count[i];
  }
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      if (sp_matrix_get(radius,x,y) < sp_min(x_max,y_max)/2.0) {
	sp_matrix_set(scale,x,y,1.0/sqrt(histogram[(int)floor(sp_matrix_get(radius,x,y))]));
      } else {
	sp_matrix_set(scale,x,y,0.0);
      }
    }
  }
  /*
  Image *out = sp_image_alloc(x_max,y_max,1);
  for (int i = 0; i < x_max*y_max; i++) {
    out->image->data[i] = sp_cinit(average->data[i],0.0);
  }
  sp_image_write(out,"debug_average.h5",0);
  for (int i = 0; i < x_max*y_max; i++) {
    out->image->data[i] = sp_cinit(scale->data[i],0.0);
  }
  sp_image_write(out,"debug_scale.h5",0);
  exit(1);
  */
}

real calculate_correlation(sp_matrix *slice, sp_matrix *image,
			   sp_matrix *average, sp_matrix *scale)
{
  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  for (int i = 0; i < i_max; i++) {
    sum += (slice->data[i] - average->data[i]) *
      (image->data[i] - average->data[i]) * pow(scale->data[i],2);
  }
  return sum;
}

/* This responsability does not yet take scaling of patterns into accoutnt. */
real calculate_responsability_absolute(sp_matrix *slice, sp_matrix *image, sp_imatrix *mask, real sigma, real scaling)
{

  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  int count = 0;
  for (int i = 0; i < i_max; i++) {
    if (mask->data[i] != 0) {
      sum += pow(slice->data[i] - image->data[i]/scaling,2);
      count++;
    }
  }
  //return exp(-sum/2.0/(real)count/pow(sigma,2));
  return -sum/2.0/(real)count/pow(sigma,2); //return in log scale.
}

real calculate_responsability_poisson(sp_matrix *slice, sp_matrix *image, sp_imatrix *mask, real sigma, real scaling)
{
  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  int count = 0;
  for (int i = 0; i < i_max; i++) {
    if (mask->data[i] != 0) {
      sum += pow(slice->data[i] - image->data[i]/scaling,2) / (1.0+image->data[i]) * scaling;
      count++;
    }
  }
  return exp(-sum/2.0/(real)count/pow(sigma,2));
}

real calculate_responsability_relative(sp_matrix *slice, sp_matrix *image, sp_imatrix *mask, real sigma, real scaling)
{
  real sum = 0.0;
  const int i_max = sp_matrix_size(slice);
  int count = 0;
  for (int i = 0; i < i_max; i++) {
    if (mask->data[i] != 0) {
      sum += pow((slice->data[i] - image->data[i]/scaling + sqrt(image->data[i]+1.0)/scaling) / (slice->data[i] + (1.0+image->data[i]/scaling)),2);
      count++;
    }
  }
  return exp(-sum/2.0/(real)count/pow(sigma,2));
}

Configuration read_configuration_file(const char *filename)
{
  Configuration config_out;
  config_t config;
  config_init(&config);
  if (!config_read_file(&config,filename)) {
    fprintf(stderr,"%d - %s\n",
	   config_error_line(&config),
	   config_error_text(&config));
    config_destroy(&config);
    exit(1);
  }
  config_lookup_int(&config,"model_side",&config_out.model_side);
  config_lookup_int(&config,"read_stride",&config_out.read_stride);
  config_lookup_float(&config,"wavelength",&config_out.wavelength);
  config_lookup_float(&config,"pixel_size",&config_out.pixel_size);
  config_lookup_int(&config,"detector_size",&config_out.detector_size);
  config_lookup_float(&config,"detector_distance",&config_out.detector_distance);
  config_lookup_int(&config,"rotations_n",&config_out.rotations_n);
  config_lookup_float(&config,"sigma",&config_out.sigma);
  config_lookup_int(&config,"slice_chunk",&config_out.slice_chunk);
  config_lookup_int(&config,"N_images",&config_out.N_images);
  config_lookup_int(&config,"max_iterations",&config_out.max_iterations);
  config_lookup_bool(&config,"blur_image",&config_out.blur_image);
  config_lookup_float(&config,"blur_sigma",&config_out.blur_sigma);
  config_lookup_string(&config,"mask_file",&config_out.mask_file);
  config_lookup_string(&config,"image_prefix",&config_out.image_prefix);
  config_lookup_bool(&config,"normalize_images",&config_out.normalize_images);
  config_lookup_bool(&config,"known_intensity",&config_out.known_intensity);
  config_lookup_int(&config,"model_input",&config_out.model_input);
  config_lookup_string(&config,"model_file",&config_out.model_file);

  return config_out;
}

sp_matrix **read_images(Configuration conf)
{
  sp_matrix **images = malloc(conf.N_images*sizeof(sp_matrix *));
  Image *img;
  real new_intensity;
  real *intensities = malloc(conf.N_images*sizeof(real));
  real scale_sum = 0.0;
  char buffer[1000];

  for (int i = 0; i < conf.N_images; i++) {
    intensities[i] = 1.0;
  }

  for (int i = 0; i < conf.N_images; i++) {
    sprintf(buffer,"%s%.4d.h5", conf.image_prefix, i);
    img = sp_image_read(buffer,0);

    /* blur image if enabled */
    if (conf.blur_image == 1) {
      Image *tmp = sp_gaussian_blur(img,conf.blur_sigma);
      sp_image_free(img);
      img = tmp;
    }

    images[i] = sp_matrix_alloc(conf.model_side,conf.model_side);
    for (int x = 0; x < conf.model_side; x++) {
      for (int y = 0; y < conf.model_side; y++) {

	sp_matrix_set(images[i],x,y,
		      sp_cabs(sp_image_get(img,
					   (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+
						 sp_image_x(img)/2-0.5),
					   (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+
						 sp_image_y(img)/2-0.5),0)));
      }
    }
    sp_image_free(img);
  }
  return images;
}

/* init mask */
sp_imatrix *read_mask(Configuration conf)
{
  sp_imatrix *mask = sp_imatrix_alloc(conf.model_side,conf.model_side);;
  Image *mask_in = sp_image_read(conf.mask_file,0);
  /* read and rescale mask */
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      if (sp_cabs(sp_image_get(mask_in,
			       (int)(conf.read_stride*((real)(x-conf.model_side/2)+0.5)+
				     sp_image_x(mask_in)/2-0.5),
			       (int)(conf.read_stride*((real)(y-conf.model_side/2)+0.5)+
				     sp_image_y(mask_in)/2-0.5),0)) == 0.0) {
	sp_imatrix_set(mask,x,y,0);
      } else {
	sp_imatrix_set(mask,x,y,1);
      }
    }
  }
  sp_image_free(mask_in);
  
  /* mask out everything outside the central sphere */
  for (int x = 0; x < conf.model_side; x++) {
    for (int y = 0; y < conf.model_side; y++) {
      if (sqrt(pow((real)x - (real)conf.model_side/2.0+0.5,2) +
	       pow((real)y - (real)conf.model_side/2.0+0.5,2)) >
	  conf.model_side/2.0) {
	sp_imatrix_set(mask,x,y,0);
      }
    }
  }
  return mask;
}

/* normalize images so average pixel value is 1.0 */
void normalize_images(sp_matrix **images, sp_imatrix *mask, Configuration conf)
{
  real sum;
  int N_2d = conf.model_side*conf.model_side;
  for (int i_image = 0; i_image < conf.N_images; i_image++) {
    sum = 0.;
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i] == 1) {
	sum += images[i_image]->data[i];
      }
    }
    sum = (real)N_2d / sum;
    for (int i = 0; i < N_2d; i++) {
      images[i_image]->data[i] *= sum;
    }
  }
}


int main(int argc, char **argv)
{
  Configuration conf;
  if (argc > 1) {
    conf = read_configuration_file(argv[1]);
  } else {
    conf = read_configuration_file("emc.conf");
  }
  const int start_iteration = 0;
  const int rescale_intensity = 0;
  const real intensity_fluct = 0.2; //when reading images they are randomly rescaled. Temporary.

  const int N_images = conf.N_images;
  const int slice_chunk = conf.slice_chunk;
  const int output_period = 5;
  const int n = conf.rotations_n;
  const int N_2d = conf.model_side*conf.model_side;
  char buffer[1000];

  Quaternion **rotations;
  real *weights;
  const long long int N_slices = generate_rotation_list(n,&rotations,&weights);
  printf("%lld rotations sampled\n",N_slices);

  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  //  gsl_rng_set(rng,time(NULL));
  // Reproducible "random" numbers
  gsl_rng_set(rng,0);

  /* read images */
  sp_matrix **images = read_images(conf);
  sp_imatrix * mask = read_mask(conf);
  if (conf.normalize_images) {
    normalize_images(images, mask, conf);
  }

  Image *write_image = sp_image_alloc(conf.model_side,conf.model_side,1);
  for (int i_image = 0; i_image < N_images; i_image++) {
    for (int i = 0; i < N_2d; i++) {
      if (mask->data[i]) {
	sp_real(write_image->image->data[i]) = images[i_image]->data[i];
      } else {
	sp_real(write_image->image->data[i]) = 0.0;
      }
    }
    sprintf(buffer, "debug/image_%.4d.png", i_image);
    sp_image_write(write_image, buffer, SpColormapJet|SpColormapLogScale);
  }

  sp_matrix *x_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *y_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  sp_matrix *z_coordinates = sp_matrix_alloc(conf.model_side,conf.model_side);
  calculate_coordinates(conf.model_side, conf.pixel_size, conf.detector_distance, conf.wavelength,
			x_coordinates, y_coordinates, z_coordinates);

  /* calculate correlation stuff */
  /*
  sp_matrix *corr_average = sp_matrix_alloc(conf.model_side, conf.model_side);
  sp_matrix *corr_scale = sp_matrix_alloc(conf.model_side, conf.model_side);
  calculate_normalization(images, N_images, corr_average, corr_scale);
  */
  /* create and fill model */
  Image *model_out = sp_image_alloc(conf.model_side,conf.model_side,conf.model_side);
  sp_3matrix *model = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  real model_d = 1.0/(conf.pixel_size*(real)conf.detector_size/conf.detector_distance*
		      conf.wavelength);
  sp_3matrix *weight = sp_3matrix_alloc(conf.model_side,conf.model_side,conf.model_side);
  const long long int N_model = conf.model_side*conf.model_side*conf.model_side;

  //change later to random rotations



  for (int i = 0; i < N_model; i++) {
    model->data[i] = 0.0;
    weight->data[i] = 0.0;
  }

  if (conf.model_input == 0) {
    printf("uniform density model\n");
    for (int i = 0; i < N_model; i++) {
      //model->data[i] = 1.0;
      model->data[i] = gsl_rng_uniform(rng);
    }
  } else if (conf.model_input == 1) {
    printf("random orientations model\n");
    Quaternion *random_rot;
    for (int i = 0; i < N_images; i++) {
      random_rot = quaternion_random(rng);
      insert_slice(model, weight, images[i], mask, 1.0, random_rot,
		   x_coordinates, y_coordinates, z_coordinates);
      free(random_rot);
    }
    for (int i = 0; i < N_model; i++) {
      if (weight->data[i] > 0.0) {
	model->data[i] /= (weight->data[i]);
      } else {
	model->data[i] = 0.0;
      }
    }
  } else if (conf.model_input == 2) {
    printf("model from file %s\n",conf.model_file);
    Image *model_in = sp_image_read(conf.model_file,0);
    if (conf.model_side != sp_image_x(model_in) ||
	conf.model_side != sp_image_y(model_in) ||
	conf.model_side != sp_image_z(model_in)) {
      printf("Input model is of wrong size.\n");
      exit(1);
    }
    for (int i = 0; i < N_model; i++) {
      model->data[i] = sp_cabs(model_in->image->data[i]);
    }
    sp_image_free(model_in);
  }

  real *scaling = malloc(N_images*sizeof(real));
  for (int i = 0; i < N_images; i++) {
    scaling[i] = 1.0;
  }

  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(model->data[i],0.0);
    if (weight->data[i] > 0.0) {
      model_out->mask->data[i] = 1;
    } else {
      model_out->mask->data[i] = 0;
    }
  }
  sprintf(buffer,"output/model_init.h5");
  sp_image_write(model_out,buffer,0);
  for (int i = 0; i < N_model; i++) {
    model_out->image->data[i] = sp_cinit(weight->data[i],0.0);
  }
  sprintf(buffer,"output/model_init_weight.h5");
  sp_image_write(model_out,buffer,0);
  printf("wrote initial model\n");

  
  /*real respons[N_slices][N_images];*/
  real *respons = malloc(N_slices*N_images*sizeof(real));
  real sum, total_respons, overal_respons, min_resp, max_resp;
  real image_power, weighted_power, correlation, scaling_error;
  real model_sum;
  FILE *likelihood = fopen("likelihood.data","wp");

  real * slices;
  cuda_allocate_slices(&slices,conf.model_side,slice_chunk); //was N_slices before
  //real * slices_on_host = malloc(N_slices*N_2d*sizeof(real));
  real * d_model;
  cuda_allocate_model(&d_model,model);
  real * d_model_updated;
  real * d_model_tmp;
  cuda_allocate_model(&d_model_updated,model);
  real * d_weight;
  cuda_allocate_model(&d_weight,weight);
  int * d_mask;
  cuda_allocate_mask(&d_mask,mask);
  real * d_rotations;
  cuda_allocate_rotations(&d_rotations,rotations,N_slices);
  real * d_x_coord;
  real * d_y_coord;
  real * d_z_coord;
  cuda_allocate_coords(&d_x_coord,
		       &d_y_coord,
		       &d_z_coord,
		       x_coordinates,
		       y_coordinates, 
		       z_coordinates);
  real * d_images;
  cuda_allocate_images(&d_images,images,N_images);
  real * d_respons;
  cuda_allocate_real(&d_respons,N_slices*N_images*sizeof(real));
  real * d_scaling;
  cuda_allocate_scaling(&d_scaling,N_images);
  real *d_weighted_power;
  cuda_allocate_real(&d_weighted_power,N_images*sizeof(real));
  real *fit = malloc(N_images*sizeof(real));
  real *d_fit;
  cuda_allocate_real(&d_fit,N_images*sizeof(real));
  int *active_images = malloc(N_images*sizeof(int));
  int *d_active_images;
  cuda_allocate_int(&d_active_images,N_images*sizeof(real));
  
  FILE *fit_file = fopen("output/fit.data","wp");
  FILE *scaling_file = fopen("output/scaling.data","wp");

  int current_chunk;
  for (int iteration = start_iteration; iteration < conf.max_iterations; iteration++) {
    sum = cuda_model_max(d_model,N_model);
    printf("model max = %g\n",sum);

    printf("iteration %d\n", iteration);
    /* get slices */
    clock_t t_i = clock();
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      if ((slice_start/slice_chunk)%output_period == 0) {
	printf("calculate presponsabilities chunk %d\n", slice_start/slice_chunk);
      }

      cuda_get_slices(model,d_model,slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);
      //cuda_copy_slice_chunk_to_host(slices_on_host, slices, slice_start, current_chunk, N_2d);


      /* calculate responsabilities */

    
      cuda_calculate_responsabilities(slices, d_images, d_mask,
				      conf.sigma, d_scaling,d_respons, 
				      N_2d, N_images, slice_start,
				      current_chunk);

    }
    printf("calculated responsabilities\n");
    clock_t t_e = clock();
    printf("Expansion time = %fs\n",(real)(t_e - t_i)/(real)CLOCKS_PER_SEC);

    cuda_calculate_responsabilities_sum(respons, d_respons, N_slices,
					N_images);
    
    printf("normalize resp\n");
    cuda_normalize_responsabilities(d_respons, N_slices, N_images);
    printf("normalize resp done\n");

    /* check how well every image fit */
    cuda_set_to_zero(d_fit,N_images);
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      cuda_get_slices(model,d_model,slices,d_rotations, d_x_coord, d_y_coord, d_z_coord,slice_start,current_chunk);
      
      cuda_calculate_fit(slices, d_images, d_mask, d_scaling,
			 d_respons, d_fit, conf.sigma, N_2d, N_images,
			 slice_start, current_chunk);
    }

    cuda_copy_real_to_host(fit, d_fit, N_images);
    for (int i_image = 0; i_image < N_images; i_image++) {
      fprintf(fit_file, "%g ", fit[i_image]);
    }
    fprintf(fit_file, "\n");
    fflush(fit_file);

    /* calculate likelihood */
    
    t_i = clock();

    total_respons = cuda_total_respons(d_respons,respons,N_images*N_slices);
    printf("calculated total resp\n");

    fprintf(likelihood,"%g\n",total_respons);
    printf("likelihood = %g\n",total_respons);
    fflush(likelihood);
  
    if (conf.known_intensity == 0) {
      /* update scaling */
      clock_t local_t_i = clock();
      cuda_set_to_zero(d_weighted_power,N_images);
      for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
	if (slice_start + slice_chunk >= N_slices) {
	  current_chunk = N_slices - slice_start;
	} else {
	  current_chunk = slice_chunk;
	}
	if ((slice_start/slice_chunk)%output_period == 0) {
	  printf("calculate weighted_power chunk %d\n", slice_start/slice_chunk);
	}

	cuda_get_slices(model, d_model, slices, d_rotations,
			d_x_coord, d_y_coord, d_z_coord,
			slice_start, current_chunk);
	//cuda_copy_slice_chunk_to_device(slices_on_host, slices, slice_start, slice_chunk, N_2d);

	cuda_update_weighted_power(d_images, slices, d_mask,
				   d_respons, d_weighted_power, N_images,
				   slice_start, current_chunk, N_2d);


      }
      cuda_update_scaling(d_images, d_mask, d_scaling, d_weighted_power,
			  N_images, N_slices, N_2d, scaling);
      clock_t local_t_e = clock();
      printf("Update scaling time = %fs\n",(real)(local_t_e - local_t_i)/(real)CLOCKS_PER_SEC);

      real scaling_sum = 0.0;
      for (int i_image = 0; i_image < N_images; i_image++) {
	scaling_sum += scaling[i_image];
      }
      scaling_sum = (real)N_images / scaling_sum;
      for (int i_image = 0; i_image < N_images; i_image++) {
	scaling[i_image] *= scaling_sum;
      }
      cuda_copy_real_to_device(scaling, d_scaling, N_images);

      if (iteration%1 == 0) {
	real average_scaling = 0.0;
	for (int i_image = 0; i_image < N_images; i_image++) {
	  average_scaling += scaling[i_image];
	}
	average_scaling /= (real)N_images;
	//printf("scaling:\n");
	real variance = 0.0;
	for (int i_image = 0; i_image < N_images; i_image++) {
	  //printf("%g, ", scaling[i_image] / average_scaling);
	  variance += pow(scaling[i_image] /average_scaling - 1.0, 2);
	}
	//printf("\n");

	variance /= (real)N_images;
	printf("scaling std = %g\n", sqrt(variance));
      }

      for (int i_image = 0; i_image < N_images; i_image++) {
	fprintf(scaling_file, "%g ", scaling[i_image]);
      }
    }
    fprintf(scaling_file, "\n");
    fflush(scaling_file);

    /* reset model */    
    cuda_reset_model(model,d_model_updated);
    cuda_reset_model(weight,d_weight);
    printf("models reset\n");


    if (iteration == 0) {
      /*
      int my_array[261] = {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
			   1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
			   0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
			   1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
			   1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			   0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
			   0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
			   1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
			   1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
			   0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,
			   1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0,
			   1, 1, 1, 1, 1, 1, 1, 1};
      for (int i_image = 0; i_image < N_images; i_image++) {
	active_images[i_image] = 1-my_array[i_image];
      }
      */

      for (int i_image = 0; i_image < N_images; i_image++) {
	active_images[i_image] = 1;
      }

    } else if (iteration == 100000) {
      for (int i_image = 0; i_image < N_images; i_image++) {
	if (log(fit[i_image]) < -10) {
	  active_images[i_image] = 1;
	} else {
	  active_images[i_image] = 0;
	}
      }
    }

    cuda_copy_int_to_device(active_images, d_active_images, N_images);

    /* update slices */
    for (int slice_start = 0; slice_start < N_slices; slice_start += slice_chunk) {
      if (slice_start + slice_chunk >= N_slices) {
	current_chunk = N_slices - slice_start;
      } else {
	current_chunk = slice_chunk;
      }
      if ((slice_start/slice_chunk)%output_period == 0) {
	printf("update slices chunk %d\n", slice_start/slice_chunk);
      }

      cuda_get_slices(model, d_model, slices, d_rotations,
		      d_x_coord, d_y_coord, d_z_coord,
		      slice_start, current_chunk);
      //cuda_copy_slice_chunk_to_device(slices_on_host, slices, slice_start, slice_chunk, N_2d);

      cuda_update_slices(d_images, slices, d_mask,
			 d_respons, d_scaling, d_active_images, N_images, slice_start, current_chunk, N_2d,
			 model,d_model_updated, d_x_coord, d_y_coord,
			 d_z_coord, &d_rotations[slice_start*4], &weights[slice_start], d_weight,images);
    }
    d_model_tmp = d_model_updated;
    d_model_updated = d_model;
    d_model = d_model_tmp;

    cuda_copy_model(model, d_model);

    printf("updated slices\n");
    t_e = clock();
    printf("Maximize time = %fs\n",(real)(t_e - t_i)/(real)CLOCKS_PER_SEC);

    t_i = clock();
    cuda_normalize_model(model, d_model,d_weight);
    printf("compressed\n");
    t_e = clock();
    printf("Compression time = %fms\n",1000.0*(t_e - t_i)/CLOCKS_PER_SEC);

    /* write output */
    for (int i = 0; i < N_model; i++) {
      model_out->image->data[i] = sp_cinit(model->data[i],0.0);
      if (weight->data[i] > 0.0) {
	model_out->mask->data[i] = 1;
      } else {
	model_out->mask->data[i] = 0;
      }
    }

    sprintf(buffer,"output/model_%.4d.h5",iteration);
    sp_image_write(model_out,buffer,0);
    printf("wrote model\n");
  }
  fclose(likelihood);
  fclose(fit_file);
  fclose(scaling_file);

}
