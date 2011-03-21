#include "rotations.h"

const real tau = (1.0 + sqrt(5.0))/2.0; // the golden ratio

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
