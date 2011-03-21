#include <spimage.h>
#include <gsl/gsl_rng.h>

typedef struct{
  real q[4];
}Quaternion;

/*
  Allocate quaternion object
 */
Quaternion *quaternion_alloc();

/*
  Make the quaternion a rotation by putting total sum of squares to 1.
 */
void quaternion_normalize(Quaternion *a);

/*
  Generate a random rotation quaternion.
 */
Quaternion *quaternion_random(gsl_rng *rng);

/*
  Calculate the number of samples for specific n.
 */
int n_to_samples(int n);

/*
  Calculate the angular sampling for specific n.
 */
real n_to_theta(int n);

/*
  Calculate the n required to sample rotationspace at a given angle.
 */
int theta_to_n(real theta);

/*
  Generate a list of rotations sampled uniformly.
 */
int generate_rotation_list(const int n, Quaternion ***return_list, real **return_weights);
