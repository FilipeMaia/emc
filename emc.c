//#include "fragmentation.h"
#include <spimage.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include "emc.h"


Quaternion *quaternion_alloc()
{
  Quaternion *res = malloc(sizeof(Quaternion));
  res->q[0] = 1.0;
  res->q[1] = 0.0;
  res->q[2] = 0.0;
  res->q[3] = 0.0;
  return res;
}

void quaternion_normalize(Quaternion *a)
{
  real abs = sqrt(pow(a->q[0],2) + pow(a->q[1],2) + pow(a->q[2],2) + pow(a->q[3],2));
  a->q[0] = a->q[0]/abs;
  a->q[1] = a->q[1]/abs;
  a->q[2] = a->q[2]/abs;
  a->q[3] = a->q[3]/abs;
}

Quaternion *quaternion_random(gsl_rng *rng)
{
  real rand1 = gsl_rng_uniform(rng);
  real rand2 = gsl_rng_uniform(rng);
  real rand3 = gsl_rng_uniform(rng);
  Quaternion *res = malloc(sizeof(Quaternion));
  res->q[0] = sqrt(1-rand1)*sin(2.0*M_PI*rand2);
  res->q[1] = sqrt(1-rand1)*cos(2.0*M_PI*rand2);
  res->q[2] = sqrt(rand1)*sin(2.0*M_PI*rand3);
  res->q[3] = sqrt(rand1)*cos(2.0*M_PI*rand3);
  return res;
}

//const real tau = (1.0 + sqrt(5.0))/2.0; // the golden ratio
const real tau = (1.0 + 2.23606798)/2.0; // the golden ratio

int n_to_samples(int n){return 20*(n+5*pow(n,3));}

real n_to_theta(int n){return 4.0 / (real) n / pow(tau,3);}

int theta_to_n(real theta){return (int) ceil(4.0 / theta / pow(tau,3));}

int generate_rotation_list(const int n, Quaternion ***return_list, real **return_weights) {
  Quaternion **rotation_list = malloc(120*sizeof(Quaternion *));

  for (int i = 0; i < 120; i++) {
    rotation_list[i] = quaternion_alloc();
    rotation_list[i]->q[0] = 0.0;
  }

  /* first 16 */
  for (int i1 = 0; i1 < 2; i1++) {
    for (int i2 = 0; i2 < 2; i2++) {
      for (int i3 = 0; i3 < 2; i3++) {
	for (int i4 = 0; i4 < 2; i4++) {
	  rotation_list[8*i1+4*i2+2*i3+i4]->q[0] = -0.5 + (real)i1;
	  rotation_list[8*i1+4*i2+2*i3+i4]->q[1] = -0.5 + (real)i2;
	  rotation_list[8*i1+4*i2+2*i3+i4]->q[2] = -0.5 + (real)i3;
	  rotation_list[8*i1+4*i2+2*i3+i4]->q[3] = -0.5 + (real)i4;
	}
      }
    }
  }
  
  /* next 8 */
  for (int i = 0; i < 8; i++) {
    rotation_list[16+i]->q[i/2] = -1.0 + 2.0*(real)(i%2);
  }

  /* last 96 */
  int it_list[12][4] = {{1,2,3,4},
		       {1,4,2,3},
		       {1,3,4,2},
		       {2,3,1,4},
		       {2,4,3,1},
		       {2,1,4,3},
		       {3,1,2,4},
		       {3,4,1,2},
		       {3,2,4,1},
		       {4,2,1,3},
		       {4,3,2,1},
		       {4,1,3,2}};

  

  for (int i = 0; i < 12; i++) {
    for (int j1 = 0; j1 < 2; j1++) {
      for (int j2 = 0; j2 < 2; j2++) {
	for (int j3 = 0; j3 < 2; j3++) {
	  rotation_list[24+8*i+4*j1+2*j2+j3]->q[it_list[i][0]-1] = -0.5 + 1.0*(real)j1;
	  rotation_list[24+8*i+4*j1+2*j2+j3]->q[it_list[i][1]-1] = tau*(-0.5 + 1.0*(real)j2);
	  rotation_list[24+8*i+4*j1+2*j2+j3]->q[it_list[i][2]-1] = 1.0/tau*(-0.5 + 1.0*(real)j3);
	  rotation_list[24+8*i+4*j1+2*j2+j3]->q[it_list[i][3]-1] = 0.0;
	}
      }
    }
  }
  
  /* get edges */
  /* all pairs of of vertices whose sum is longer than 3 is an edge */
  FILE *f = fopen("debug_edges.data","wp");
  real dist2;
  int count = 0.0;
  real edge_cutoff = 3.0;

  int edges[720][2];
  for (int i = 0; i < 120; i++) {
    for (int j = 0; j < i; j++) {
      dist2 =
	pow(rotation_list[i]->q[0] + rotation_list[j]->q[0],2) +
	pow(rotation_list[i]->q[1] + rotation_list[j]->q[1],2) +
	pow(rotation_list[i]->q[2] + rotation_list[j]->q[2],2) +
	pow(rotation_list[i]->q[3] + rotation_list[j]->q[3],2);
      if (dist2 > edge_cutoff) {
	edges[count][0] = i;
	edges[count][1] = j;
	count++;
	fprintf(f,"%d %d %g\n",i,j,sqrt(dist2));
      }
    }
  }
  printf("%d edges\n",count);
  fclose(f);

  /* get faces */
  /* all pairs of edge and vertice whith a sum larger than 7.5 is a face */
  real face_cutoff = 7.5;
  int face_done[120];
  for (int i = 0; i < 120; i++) {face_done[i] = 0;}
  count = 0;
  int faces[1200][3];
  for (int i = 0; i < 720; i++) {
    face_done[edges[i][0]] = 1;
    face_done[edges[i][1]] = 1;
    for (int j = 0; j < 120; j++) {
      //if (edges[i][0] == j || edges[i][1] == j) {
	/* continue if the vertex is already in the edge */
	//continue;
      //}
      if (face_done[j]) {
	/* continue if the face has already been in a vertex,
	   including the current one */
	continue;
      }
      dist2 =
	pow(rotation_list[j]->q[0] + rotation_list[edges[i][0]]->q[0] +
	    rotation_list[edges[i][1]]->q[0], 2) +
	pow(rotation_list[j]->q[1] + rotation_list[edges[i][0]]->q[1] +
	    rotation_list[edges[i][1]]->q[1], 2) +
	pow(rotation_list[j]->q[2] + rotation_list[edges[i][0]]->q[2] +
	    rotation_list[edges[i][1]]->q[2], 2) +
	pow(rotation_list[j]->q[3] + rotation_list[edges[i][0]]->q[3] +
	    rotation_list[edges[i][1]]->q[3], 2);
      if (dist2 > face_cutoff) {
	faces[count][0] = edges[i][0];
	faces[count][1] = edges[i][1];
	faces[count][2] = j;
	count++;
      }
    }
  }
  printf("%d faces\n",count);

  /* get cells */
  /* all pairs of face and vertice with a sum larger than 13.5 is a cell */

  real cell_cutoff = 13.5;
  int cell_done[120];
  for (int i = 0; i < 120; i++) {cell_done[i] = 0;}
  count = 0;
  int cells[600][4];
  for (int j = 0; j < 120; j++) {
    cell_done[j] = 1;
    for (int i = 0; i < 1200; i++) {
      /*if (cell_done[j]) {
	continue;
	}*/
      if (cell_done[faces[i][0]] || cell_done[faces[i][1]] || cell_done[faces[i][2]]) {
	continue;
      }
      /*
      if (faces[i][0] == j || faces[i][1] == j || faces[i][2] == j) {
	continue;
      }
      */
      dist2 =
	pow(rotation_list[faces[i][0]]->q[0] + rotation_list[faces[i][1]]->q[0] +
	    rotation_list[faces[i][2]]->q[0] + rotation_list[j]->q[0], 2) +
	pow(rotation_list[faces[i][0]]->q[1] + rotation_list[faces[i][1]]->q[1] +
	    rotation_list[faces[i][2]]->q[1] + rotation_list[j]->q[1], 2) +
	pow(rotation_list[faces[i][0]]->q[2] + rotation_list[faces[i][1]]->q[2] +
	    rotation_list[faces[i][2]]->q[2] + rotation_list[j]->q[2], 2) +
	pow(rotation_list[faces[i][0]]->q[3] + rotation_list[faces[i][1]]->q[3] +
	    rotation_list[faces[i][2]]->q[3] + rotation_list[j]->q[3], 2);
      if (dist2 > cell_cutoff) {
	cells[count][0] = faces[i][0];
	cells[count][1] = faces[i][1];
	cells[count][2] = faces[i][2];
	cells[count][3] = j;
	count++;
      }
    }
  }
  printf("%d cells\n",count);

  /*variables used to calculate the weights */
  real alpha = acos(1.0/3.0);
  real f1 = 5.0*alpha/2.0/M_PI;
  real f0 = 20.0*(3.0*alpha-M_PI)/4.0/M_PI;
  real f2 = 1.0;
  real f3 = 1.0;

  int number_of_samples = n_to_samples(n);
  printf("%d samples\n",number_of_samples);
  Quaternion **new_list = malloc(number_of_samples*sizeof(Quaternion *));
  for (int i = 0; i < number_of_samples; i++) {
    new_list[i] = quaternion_alloc();
  }

  real *weights = malloc(number_of_samples*sizeof(real));
  real dist3;

  /* copy vertices */
  for (int i = 0; i < 120; i++) {
    new_list[i]->q[0] = rotation_list[i]->q[0];
    new_list[i]->q[1] = rotation_list[i]->q[1];
    new_list[i]->q[2] = rotation_list[i]->q[2];
    new_list[i]->q[3] = rotation_list[i]->q[3];
    dist3 = pow(pow(new_list[i]->q[0],2)+
		pow(new_list[i]->q[1],2)+
		pow(new_list[i]->q[2],2)+
		pow(new_list[i]->q[3],2),(real)3/(real)2);
    weights[i] = f0/(real)number_of_samples/dist3;
  }

  /* split edges */
  int edges_base = 120;
  int edge_verts = (n-1);
  int index;
  printf("edge_verts = %d\n",edge_verts);
  for (int i = 0; i < 720; i++) {
    for (int j = 0; j < edge_verts; j++) {
      index = edges_base+edge_verts*i+j;
      for (int k = 0; k < 4; k++) {
	new_list[index]->q[k] = 
	  (real)(j+1) / (real)(edge_verts+1) * rotation_list[edges[i][0]]->q[k] +
	  (real)(edge_verts-j) / (real)(edge_verts+1) * rotation_list[edges[i][1]]->q[k];
      }
      dist3 = pow(pow(new_list[index]->q[0],2) + pow(new_list[index]->q[1],2) +
		  pow(new_list[index]->q[2],2) + pow(new_list[index]->q[3],2), (real)3/(real)2);
      weights[index] = f1/(real)number_of_samples/dist3;
    }
  }

  /* split faces */
  int faces_base = 120 + 720*edge_verts;
  int face_verts = ((n-1)*(n-2))/2;
  real a,b,c;
  int kc;
  printf("face_verts = %d\n",face_verts);
  if (face_verts > 0) {
    for (int i = 0; i < 1200; i++) {
      count = 0;
      for (int ka = 2; ka < edge_verts+1; ka++) {
	for (int kb = 2; kb < edge_verts+1; kb++) {
	  if (ka + kb > edge_verts+1) {
	    kc = 2*(edge_verts+1)-ka-kb;
	    a = (real) (edge_verts + 1 - ka) / (real) (3*(edge_verts+1)-ka-kb-kc);
	    b = (real) (edge_verts + 1 - kb) / (real) (3*(edge_verts+1)-ka-kb-kc);
	    c = (real) (edge_verts + 1 - kc) / (real) (3*(edge_verts+1)-ka-kb-kc);
	    index = faces_base+face_verts*i+count;
	    for (int k = 0; k < 4; k++) {
	      new_list[index]->q[k] =
		a * rotation_list[faces[i][0]]->q[k] +
		b * rotation_list[faces[i][1]]->q[k] +
		c * rotation_list[faces[i][2]]->q[k];
	    }
	    //printf("k1 = %d\nkb = %d\nkc = %d\n",ka,kb,kc);
	    //printf("a = %g\nb = %g\nc = %g\n",a,b,c);
	    dist3 = pow(pow(new_list[index]->q[0],2) + pow(new_list[index]->q[1],2) +
			pow(new_list[index]->q[2],2) + pow(new_list[index]->q[3],2), (real)3/(real)2);
	    weights[index] = f2/(real)number_of_samples/dist3;
	    count++;
	  }
	}
      }
    }
  }

  /* split cells */
  int cell_base = 120 + 720*edge_verts + 1200*face_verts;
  int cell_verts = ((n-1)*(n-2)*(n-3))/6;
  real d;
  int kd;
  printf("cell_verts = %d\n",cell_verts);
  int debug_count = 0;
  if (cell_verts > 0) {
    for (int i = 0; i < 600; i++) { //600
      count = 0;
      for (int ka = 3; ka < edge_verts+1; ka++) {
	for (int kb = 3; kb < edge_verts+1; kb++) {
	  for (int kc = 3; kc < edge_verts+1; kc++) {
	    kd = 3*(edge_verts+1)-ka-kb-kc;
	    if (kd >= 3 && kd < edge_verts+1) {
	      a = (real) (edge_verts + 1 - ka) / (real) (4*(edge_verts+1)-ka-kb-kc-kd);
	      b = (real) (edge_verts + 1 - kb) / (real) (4*(edge_verts+1)-ka-kb-kc-kd);
	      c = (real) (edge_verts + 1 - kc) / (real) (4*(edge_verts+1)-ka-kb-kc-kd);
	      d = (real) (edge_verts + 1 - kd) / (real) (4*(edge_verts+1)-ka-kb-kc-kd);
	      index = cell_base+cell_verts*i+count;
	      for (int k = 0; k < 4; k++) {
		new_list[index]->q[k] =
		  a*rotation_list[cells[i][0]]->q[k] +
		  b*rotation_list[cells[i][1]]->q[k] +
		  c*rotation_list[cells[i][2]]->q[k] +
		  d*rotation_list[cells[i][3]]->q[k];
	      }
	      dist3 = pow(pow(new_list[index]->q[0],2) + pow(new_list[index]->q[1],2) +
			  pow(new_list[index]->q[2],2) + pow(new_list[index]->q[3],2), (real)3/(real)2);
	      weights[index] = f3/(real)number_of_samples/dist3;
	      count++;
	      debug_count++;
	      //printf("\na = %g\nb = %g\nc = %g\nd = %g\n",a,b,c,d);
	    }
	  }
	}
      }
    }
  }
  printf("debug_count = %d\n",debug_count);

  real weight_sum = 0.0;
  //for (int i = edges_base; i < edges_base + edge_verts*720; i++) {
  for (int i = 0; i < number_of_samples; i++) {
    weight_sum += weights[i];
  }
  printf("weights sum = %g\n",weight_sum);
  for (int i = 0; i < number_of_samples; i++) {
    weights[i] /= weight_sum;
  }
  
  for (int i = 0; i < 120; i++) {
    free(rotation_list[i]);
  }
  free(rotation_list);

  return_list[0] = new_list;
  return_weights[0] = weights;

  f =  fopen("debug_samples.data","wp");
  for (int i = 0; i < number_of_samples; i++) {
    quaternion_normalize(new_list[i]);
    fprintf(f,"%g %g %g %g\n",new_list[i]->q[0],new_list[i]->q[1],new_list[i]->q[2],new_list[i]->q[3]);
  }
  fclose(f);

  return number_of_samples;
}

void calculate_coordinates(Setup setup, sp_matrix *x_coordinates,
			   sp_matrix *y_coordinates, sp_matrix *z_coordinates) {
  const int x_max = setup.side;
  const int y_max = setup.side;
  real pixel_r, real_r, fourier_r, angle_r, fourier_z;
  real pixel_x, pixel_y, pixel_z;
  //tabulate angle later
  for (int x = 0; x < x_max; x++) {
    for (int y = 0; y < y_max; y++) {
      pixel_r = sqrt(pow((real)(x-x_max/2)+0.5,2) + pow((real)(y-y_max/2)+0.5,2));
      real_r = pixel_r*setup.pixel_size;
      angle_r = atan2(real_r,setup.detector_distance);
      fourier_r = sin(angle_r)/setup.wavelength;
      fourier_z = (1. - cos(angle_r))/setup.wavelength;
      /*
      angle_x = atan2((real)(x-xmax/2),setup->detector_distance);
      angle_y = atan2((real)(x-xmax/2),setup->detector_distance);
      fourier_x = cos(angle_x)/setup->wavelength;
      fourier_y = cos(angle_y)/setup->wavelength;
      */
      pixel_x = (real)(x-x_max/2)+0.5;
      pixel_y = (real)(y-y_max/2)+0.5;
      pixel_z = fourier_z/fourier_r*pixel_r;
      sp_matrix_set(x_coordinates,x,y,pixel_x);
      sp_matrix_set(y_coordinates,x,y,pixel_y);
      sp_matrix_set(z_coordinates,x,y,pixel_z);
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

int main(int argc, char **argv)
{
  const int model_input = 0; //0 = random_density, 1 = random_orientations, 2 = from_file
  const char *model_file = "model_0014.h5";
  const int start_iteration = 0;
  const int known_intensity = 0;
  const int rescale_intensity = 0;
  const real intensity_fluct = 0.2; //when reading images they are randomly rescaled. Temporary.
  const int mask_input = 1; //0 = no_mask, 1 = mask_from_file
  //const char *mask_file = "/home/ekeberg/Work/max/spowpy/virus_dataset/mask.h5";
  const char *mask_file = "mask.h5";

  const int blur_image = 0; //0 = no, 1 = yes
  const real blur_radius = 3.0;

  const int max_iterations = 1000;
  /* 0.03 with absolute responsability type.
     0.4 absolute resp, stride=3, side=64
     0.004 with relative responsability type.
     0.04 with poisson responsability type.
     1.0 absolute resp, stride=3, side=64
   */
  const real sigma = 0.4; //0.4 is to low.
  const real start_sigma = 0.4;
  const int N_images = 100;
  //  const int N_images = 100;
 
  real (*calculate_responsability)(sp_matrix *, sp_matrix *, sp_imatrix *, real, real) =
    &calculate_responsability_absolute;

  gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
  //  gsl_rng_set(rng,time(NULL));
  // Reproducible "random" numbers
  gsl_rng_set(rng,0);
  /*
    Directions *reference = directions_random(num_ions,rng);
    Quaternion **true_rotations = malloc(num_explosions*sizeof(Quaternion *));
    Dirset *set = dirset_random(reference, num_explosions, noise, rng, true_rotations);
  */


  const int n = 4;
  Quaternion **rotations;
  real *weights;
  const long long int N_slices = generate_rotation_list(n,&rotations,&weights);
  printf("%lld rotations sampled\n",N_slices);
  
  // create density map from random orientations
  // iterate
  // take out orientations from densitymap
  // calculate similarities between images and slices
  // puth together new densitymap from images
  // end iterate

  Setup setup;
  setup.side = 64;
  setup.wavelength = 5.6e-9;
  setup.pixel_size = 16e-6;
  setup.detector_size = 512;
  setup.detector_distance = 0.15;

  char buffer[1000];

  const int N_2d = setup.side*setup.side;

  /* read images */
  const int read_stride = 4; //should be even
  sp_matrix **images = malloc(N_images*sizeof(sp_matrix *));
  Image *img;
  real new_intensity;
  real *intensities = malloc(N_images*sizeof(real));
  real scale_sum = 0.0;

  for (int i = 0; i < N_images; i++) {
    if (rescale_intensity == 1) {
      intensities[i] = 1.0 + intensity_fluct*(-1.0+2.0*gsl_rng_uniform(rng));
    } else {
      intensities[i] = 1.0;
    }
    scale_sum += intensities[i];
  }
  scale_sum /= (real)N_images;
  for (int i = 0; i < N_images; i++) {
    intensities[i] /= scale_sum;
  }

  printf("new intensities: ");
  for (int i = 0; i < N_images; i++) {
    printf("%g ", intensities[i]);
    //sprintf(buffer,"/home/ekeberg/Work/max/spowpy/virus_dataset/theoretical/image_%.4d.h5",i);
    //sprintf(buffer,"/home/ekeberg/Work/max/spowpy/virus_dataset/poisson/image_%.4d.h5",i);
    sprintf(buffer,"Data/image_%.4d.h5",i);
    img = sp_image_read(buffer,0);
    if (blur_image == 1) {
      Image *tmp = sp_gaussian_blur(img,blur_radius);
      sp_image_free(img);
      img = tmp;
    }
    //images[i] = sp_matrix_alloc(sp_image_x(img),sp_image_y(img));
    images[i] = sp_matrix_alloc(setup.side,setup.side);
    for (int x = 0; x < setup.side; x++) {
      for (int y = 0; y < setup.side; y++) {

	sp_matrix_set(images[i],x,y,intensities[i] *
		      sp_cabs(sp_image_get(img,(int)(read_stride*((real)(x-setup.side/2)+0.5)+
						     sp_image_x(img)/2-0.5),
					   (int)(read_stride*((real)(y-setup.side/2)+0.5)+
						 sp_image_y(img)/2-0.5),0)));
	/*
	if (x > setup.side/2) {
	  sp_matrix_set(images[i],x,y,1.0);
	} else {
	  sp_matrix_set(images[i],x,y,0.1);
	}
	*/
      }
    }
    sp_image_free(img);
  }
  printf("\n");

  /* init mask */
  sp_imatrix *mask = sp_imatrix_alloc(setup.side,setup.side);;
  if (mask_input == 0) {
    for (int i = 0; i < N_2d; i++) {
      mask->data[i] = 1;
    }
  } else if (mask_input == 1) {
    Image *mask_in = sp_image_read(mask_file,0);
    for (int x = 0; x < setup.side; x++) {
      for (int y = 0; y < setup.side; y++) {
	if (sp_cabs(sp_image_get(mask_in,
						    (int)(read_stride*((real)(x-setup.side/2)+0.5)+
							  sp_image_x(mask_in)/2-0.5),
						    (int)(read_stride*((real)(y-setup.side/2)+0.5)+
							  sp_image_y(mask_in)/2-0.5),0)) == 0.0) {
	  sp_imatrix_set(mask,x,y,0);
	} else {
	  sp_imatrix_set(mask,x,y,1);
	}
      }
    }
    sp_image_free(mask_in);
  }
  for (int x = 0; x < setup.side; x++) {
    for (int y = 0; y < setup.side; y++) {
      if (sqrt(pow((real)x - (real)setup.side/2.0+0.5,2) +
	       pow((real)y - (real)setup.side/2.0+0.5,2)) >
	  setup.side/2.0) {
	sp_imatrix_set(mask,x,y,0);
      }
    }
  }

  /* calculate correlation stuff */
  /*
  sp_matrix *corr_average = sp_matrix_alloc(setup.side, setup.side);
  sp_matrix *corr_scale = sp_matrix_alloc(setup.side, setup.side);
  calculate_normalization(images, N_images, corr_average, corr_scale);
  */
  sp_matrix *x_coordinates = sp_matrix_alloc(setup.side,setup.side);
  sp_matrix *y_coordinates = sp_matrix_alloc(setup.side,setup.side);
  sp_matrix *z_coordinates = sp_matrix_alloc(setup.side,setup.side);
  calculate_coordinates(setup, x_coordinates, y_coordinates, z_coordinates);

  /* create and fill model */
  Image *model_out = sp_image_alloc(setup.side,setup.side,setup.side);
  sp_3matrix *model = sp_3matrix_alloc(setup.side,setup.side,setup.side);
  real model_d = 1.0/(setup.pixel_size*(real)setup.detector_size/setup.detector_distance*
		      setup.wavelength);
  sp_3matrix *weight = sp_3matrix_alloc(setup.side,setup.side,setup.side);
  const long long int N_model = setup.side*setup.side*setup.side;

  //change later to random rotations



  for (int i = 0; i < N_model; i++) {
    model->data[i] = 0.0;
    weight->data[i] = 0.0;
  }

  if (model_input == 0) {
    printf("uniform density model\n");
    for (int i = 0; i < N_model; i++) {
      //model->data[i] = 1.0;
      model->data[i] = gsl_rng_uniform(rng);
    }
  } else if (model_input == 1) {
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
  } else if (model_input == 2) {
    printf("model from file %s\n",model_file);
    Image *model_in = sp_image_read(model_file,0);
    if (setup.side != sp_image_x(model_in) ||
	setup.side != sp_image_y(model_in) ||
	setup.side != sp_image_z(model_in)) {
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

  /* alloc slices */
  sp_matrix **slices = malloc(N_slices*sizeof(sp_matrix *));
  for (int i = 0; i < N_slices; i++) {
    slices[i] = sp_matrix_alloc(setup.side,setup.side);
  }
  
  /*real respons[N_slices][N_images];*/
  real *respons = malloc(N_slices*N_images*sizeof(real));
  real sum, total_respons, overal_respons, min_resp, max_resp;
  real image_power, weighted_power, correlation, scaling_error;
  real model_sum;
  FILE *scale = fopen("scale_error.data","wp");
  FILE *likelihood = fopen("likelihood.data","wp");
  for (int iteration = start_iteration; iteration < max_iterations; iteration++) {
    sum = 0.0;
    for (int i = 0; i < N_model; i++) {
      if (model->data[i] > sum) {
	sum = model->data[i];
      }
    }
    printf("model max = %g\n",sum);

    printf("iteration %d\n", iteration);
    /* get slices */
    for (int i = 0; i < N_slices; i++) {
      get_slice(model, slices[i], rotations[i], x_coordinates, y_coordinates, z_coordinates);
    }
    /*
    Image *foo = sp_image_alloc(setup.side,setup.side,1);
    for (int i = 0; i < N_2d; i++) {
      foo->image->data[i] = sp_cinit(slices[0]->data[i],0.0);
    }
    sp_image_write(foo,"debug_first_slice.h5",0);
    exit(1);
    */

    printf("expanded\n");
    /* calculate responsabilities */
    /*
    sprintf(buffer,"debug_respons_%d.data",iteration);
    f = fopen(buffer,"wp");
    */
    clock_t t_i = clock();
    if(iteration != 0){
      cuda_calculate_responsabilities(slices, images, mask,
				      sigma, scaling,respons, 
				      N_2d, N_images, N_slices);
    }else{
      cuda_calculate_responsabilities(slices, images, mask,
				      start_sigma, scaling,respons, 
				      N_2d, N_images, N_slices);

    }
    for (int i_image = 0; i_image < N_images; i_image++) {
      sum = 0.0;
      min_resp = 1.0;
      max_resp = -1.0e10;
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	if (max_resp < respons[i_slice*N_images+i_image]) {
	  max_resp = respons[i_slice*N_images+i_image];
	}
      }
      //printf("max_resp = %g\n",max_resp);
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	respons[i_slice*N_images+i_image] -= max_resp;
      }
      sum = 0.0;
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	if (respons[i_slice*N_images+i_image] > -1.0e10) {
	  respons[i_slice*N_images+i_image] = exp(respons[i_slice*N_images+i_image]);
	  sum += respons[i_slice*N_images+i_image];
	} else {
	  respons[i_slice*N_images+i_image] = 0.0;
	}
      }
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	respons[i_slice*N_images+i_image] /= sum;
      }
    }

    clock_t t_e = clock();
    printf("Expansion time = %fms\n",1000.0*(t_e - t_i)/CLOCKS_PER_SEC);
    /* calculate likelihood */
    
    t_i = clock();
    total_respons = 0.0;
    for (int i_image = 0; i_image < N_images; i_image++) {
      for (int i_slice = 0; i_slice < N_slices; i_slice++) {
	total_respons += respons[i_slice*N_images+i_image]*log(respons[i_slice*N_images+i_image]);
      }
    }
    fprintf(likelihood,"%g\n",total_respons);
    printf("likelihood = %g\n",total_respons);
    fflush(likelihood);
  
    printf("calculated responsabilities\n");
    /* reset model */
    for (int i = 0; i < N_model; i++) {
      model->data[i] = 0.0;
      weight->data[i] = 0.0;
    }
    clock_t local_t_i = clock();
    /* update scaling */
    if (known_intensity == 0) {
      scaling_error = 0.0;
      cuda_update_scaling(images, slices, mask,
			  respons, scaling, N_images, N_slices, N_2d);     
      scale_sum = 0.0;
      for (int i = 0; i < N_images; i++) {
	scale_sum += scaling[i];
      }
      scale_sum /= (real)N_images;

      /*
      for (int i = 0; i < N_images; i++) {
	scaling[i] /= N_images;
      }
      */
      
      if (iteration % 10 == 0) {printf("new scaling: ");}
      correlation = 0.0;
      for (int i = 0; i < N_images; i++) {
	correlation += pow(scaling[i]/scale_sum - intensities[i],2);
	if (iteration % 10 == 0) {printf("%g, ",scaling[i]/scale_sum);}
      }
      if (iteration % 10 == 0) {printf("\n");}
      correlation = sqrt(correlation/(real)N_images);
      printf("scaling error = %g\n",correlation);
      fprintf(scale,"%g\n",correlation);
      fflush(scale);
      
    }
    clock_t local_t_e = clock();
        printf("Update scaling time = %fms\n",1000.0*(local_t_e - local_t_i)/CLOCKS_PER_SEC);
    /* update slices */
    overal_respons = cuda_update_slices(images, slices, mask,
					respons, scaling, N_images, N_slices, N_2d,
					model, x_coordinates, y_coordinates,
					z_coordinates, rotations, weights, weight);

    t_e = clock();
    printf("Maximize time = %fms\n",1000.0*(t_e - t_i)/CLOCKS_PER_SEC);

    t_i = clock();
    model_sum = 0.0;
    for (int i = 0; i < N_model; i++) {
      if (weight->data[i] > 0.0) {
	model->data[i] /= (weight->data[i]);
	model_sum += model->data[i];
      } else {
	model->data[i] = 0.0;
      }
    }
    model_sum /= (real)N_model;
    for (int i = 0; i < N_model; i++) {
      model->data[i] /= model_sum;
    }
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
  fclose(scale);
  fclose(likelihood);

}
