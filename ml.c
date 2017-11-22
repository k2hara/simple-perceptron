/* linear perceptron learning program.        */
/* Teacher and studnet are linear perceptron. */
/* Learning algorithm is gradient descent.    */
/* Compile: gcc -o ml ml.c -lm                */
/* Run: ml                                    */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "SFMT.c"
#include "SFMT.h"

#define NN 1000   /* input dimension */

void gen_data(double *x);                                      /* data generator */
void init_weight(double *w);                                   /* Teacher vector generator */
void gd(double *w, double v, double u, double *x, double eta); /* gradient descent rule */
double inner_p(double *b, double *j);                          /* calculate Inner Product */
double norm(double *p);                                        /* calculate norm of vector */
double length(double *w);                                      /* calculate length of vector */ 
double overlap(double *b, double *j);                          /* calculate overlap of two vectors */
double eg(double *b, double *j);                               /* Generalization error using R and l */
double gasdev(void);                                   /* Gaussian rundom number generator */

int main()
{
  double x[NN];              /* data */
  double b[NN];              /* teacher network */
  double j[NN];              /* student network */
  double v;                 /* teacher output */
  double u;                 /* ensemble output */
  double err;               /* output error */
  double R;                 /* overlap */
  double eta;               /* learning rate */
  int i;                    /* roop valiable */
  int it;                   /* iteration number */
  long seed;

  eta=0.1;                  /* initialize learning rate */
  seed=1L;            /* initialize seed */
  it=100000;                /* initialize iteration number */
  init_gen_rand(seed);
  init_weight(b);           /* initialize teacher weight vector */
  init_weight(j);           /* initialize student weight vector */

  for(i=0;i<it;i++){
    if(i%NN==0){  /* print results */
      printf("%lf ",(double)i/NN);                 
      printf("%lf %lf %lf \n",length(j),overlap(b,j),eg(b,j)); 
    }
    gen_data(x);                     /* generate input */
    v=inner_p(x,b);                  /* calculate teacher output */
    u=inner_p(x,j);                  /* calculate student output */
    gd(j,v,u,x,eta);                 /* update student weight vector */
  }
  return 0;
}

void gen_data(double *x)
/* Data generator */
{
  int i;

  for(i=0;i<NN;i++){
    x[i]=gasdev()/sqrt(NN); /* initalize data by Gaussian random number */
  }
}

void init_weight(double *w)
/* Initialize teacher weight vector */
{
  int i;

  for(i=0;i<NN;i++){
    w[i]=gasdev();
  }
}

void gd(double *w, double v, double u, double *x, double eta)
/* gradient descent rule */
{
  int i;

  for(i=0;i<NN;i++){
    w[i]=w[i]+eta*(v-u)*x[i];
  }
}

double inner_p(double *b, double *j)
{
  int i;
  double in;

  in=0.0;
  for(i=0;i<NN;i++){
    in+=b[i]*j[i];
  }
  return in;
}

double norm(double *p)
{
  int i;
  double q;

  q=0.0;
  for(i=0;i<NN;i++){
    q+=p[i]*p[i];
  }
  return sqrt(q);
}

double length(double *w)
{
  return norm(w)/sqrt(NN);
}

double overlap(double *b, double *j)
{
  int i;

  return inner_p(b,j)/(norm(b)*norm(j));
}

double eg(double *b, double *j)
/* Generalization error of linear perceptron using data */
{
  int i;
  double x[NN], v, u;
  double eg;

  eg=0.0;
  for(i=0;i<NN;i++){
    gen_data(x);
    v=inner_p(x,b);    /* input of teacher net */
    u=inner_p(x,j);    /* input of student net */
    eg+=(u-v)*(u-v)/2.0;
  }
  eg/=NN;
  return eg;
}

/* Gaussian random number generation part */
/* From Numerical recipy in C */
double gasdev(void)
{
  static int iset=0;
  static double gset;
  double fac,rsq,v1,v2;

  if(iset==0){
    do{
      v1=2.0*genrand_res53()-1.0;
      v2=2.0*genrand_res53()-1.0;
      rsq=v1*v1+v2*v2;
    }while (rsq >= 1.0 || rsq == 0.0);
    fac=sqrt(-2.0*log(rsq)/rsq);
    gset=v1*fac;
    iset=1;
    return v2*fac;
  }else{
    iset=0;
  }
    return gset;
}

