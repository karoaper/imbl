/*
 * =====================================================================================
 * 
 *        Filename:  ibl.c
 * 
 *     Description:  The C source code for the IBL interface.  Contains the algorithmic functions, for traning, predicting, doing cross-validation, etc.  Prototypes are defined in ibl.h 
 *		     This version is different than others in that, the code hasbeen cleaned, functions that we think will not be used any more have been removed, rather than just commented.  
 *		     Also, starting from this version, we will try to keep a better track of changes and revisions.
 *
 * 
 *         Version:  2.0
 *         Created:  02/23/2007 04:05:39 PM MST
 *        Revision:  1.0.0
 *        Compiler:  gcc
 * 
 *          Author:  Karen Hovsepian 
 *         Company:  New Mexico Tech
 * 
 * =====================================================================================
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include "imbl.h"
#include <time.h>
#include "acml.h"

#define min(a,b)	(a < b ? a : b)
#define max(a,b)	(a < b ? b : a)

#define rbffun(dist,sigma)  exp(dist*sigma)
#define atanfun(dist,sigma)  1-2/M_PI*atan(dist*sigma)

struct imbl_model * train(struct learning_problem *problem, double sigma)
{
	int i,j;
	struct imbl_model *model = (struct imbl_model *) malloc(sizeof(struct imbl_model));

	model->sigma = sigma;

	int l = problem->l;

	model->l = l;
	struct vector_node ** x;
	double *K = Malloc(double,l*l);
	double *U = Malloc(double,l*l);
	double *B = Malloc(double,l);
	int start [2];//= Malloc(int,nr_class+1);
	int *count = problem->count;
//	model->count_class_one = count[0];

	int *perm = Malloc(int,l);

	//------begin group classes

	start[0] = 0;
	start[1] = problem->count[0];

	for(i=0;i<l;i++)
	{
		perm[start[problem->y[i]]] = i;
		++start[problem->y[i]];
	}

	//---------end group classes

	x = Malloc(struct vector_node *,l);
	
	for(i=0;i<l;i++)
	{
		x[i] = problem->x[perm[i]];
	}

#ifdef NORM_EUCLID
	double * variances_overall = get_variances(x,0,l,problem->dim);
	double * variances_c1 = get_variances(x,0,count[0],problem->dim);
	double * variances_c2 = get_variances(x,count[0],count[1],problem->dim);
#endif

	model->count = count;
	double * spread_euclid = Malloc(double,2);
	model->x = x;
    	int Nm = max(problem->count[0],problem->count[1]);
	int Nmax = 2*Nm;
	
	
	double b = Nm + 0.5;
	spread_euclid[0] = 0;
	for(i=0;i<count[0];i++)
	{
		B[i] = b;
		K(i,i) = Nmax;
			
		for(j=i+1;j<count[0];j++)
		{
#ifdef NORM_EUCLID
			double temp = distance2(x[j],x[i],variances_c1);
#else
			double temp = distance1(x[j],x[i]);
#endif
			spread_euclid[0] += temp;
			U(i,j) = temp;
		}
			
		for(j=count[0];j<l;j++)
		{
#ifdef NORM_EUCLID
			K(i,j) = kernel2(x[j],x[i],sigma,variances_overall);
#else
			K(i,j) = kernel1(x[j],x[i],sigma);
#endif
		}
	}

	spread_euclid[1] = 0;
	for(i=count[0];i<l;i++)
	{
		B[i] = b;
		K(i,i) = Nmax;
			
		for(j=i+1;j<l;j++)
		{
#ifdef NORM_EUCLID
			double temp = distance2(x[j],x[i],variances_c2);
#else
			double temp = distance1(x[j],x[i]);
#endif
			spread_euclid[1] += temp;	
			U(i,j) = temp;
		}
	}
	model->spread_euclid = spread_euclid;	

	double temp_own = (count[1]*(count[1]-1)*spread_euclid[0])/(count[0]*(count[0]-1)*spread_euclid[1]);
	
	if(temp_own > 1)
	{
		for(i=0;i<count[0];i++)
		{
			for(j=i+1;j<count[0];j++)
			{
#ifdef RBF
				K(i,j) = -rbffun(-U(i,j),sigma);
#else	
				K(i,j) = -atanfun(U(i,j),sigma);
#endif
			}
		}
		temp_own *= sigma;
		for(i=count[0];i<l-1;i++)
		{
			for(j=i+1;j<l;j++)
			{
#ifdef RBF
				K(i,j) = -rbffun(-U(i,j),temp_own);
#else	
				K(i,j) = -atanfun(U(i,j),temp_own);
#endif
			}
		}
	}	
	else
	{
		temp_own = sigma/temp_own;
		for(i=0;i<count[0];i++)
		{
			for(j=i+1;j<count[0];j++)
			{
#ifdef RBF
				K(i,j) = -rbffun(-U(i,j),temp_own);
#else	
				K(i,j) = -atanfun(U(i,j),temp_own);
#endif
			}
		}
		for(i=count[0];i<l-1;i++)
		{
			for(j=i+1;j<l;j++)
			{
#ifdef RBF
				K(i,j) = -rbffun(-U(i,j),sigma);
#else	
				K(i,j) = -atanfun(U(i,j),sigma);
#endif
			}
		}
	}
	dpotrf('U', l,K, l, &i);
	dpotrs ('U', l, 1, K, l,B, l, &i);
	model->alpha = B;
	free(K);
	free(U);
	free(perm);
	return model;
}


int predict(struct imbl_model *model,const struct vector_node *x_test)
{
    int j;
    struct vector_node **x = model->x;
    double sigma = model->sigma;
    double *alpha = model->alpha;
    int *count = model->count;	
	double tempS [2] = {0,0};

	int l = model->l;
	double *s_euclid = Malloc(double,l);   //to keep the linear distances of class 1, assuming x is in class 1, for later use
	double distance_euclid = model->spread_euclid[0];

	for(j=0;j<count[0];j++)
	{
		s_euclid[j] = distance1(x_test,x[j]);  
		distance_euclid += s_euclid[j];    //sum of linear distances of x to class 0
#ifdef RBF
		tempS[0] += alpha[j]*rbffun(-s_euclid[j],sigma);
#else
		tempS[0] += alpha[j]*atanfun(s_euclid[j],sigma);
#endif
	}	

	double temp_own = (distance_euclid*count[1]*(count[1]-1))/(count[0]*(count[0]+1)*model->spread_euclid[1]);
        if(temp_own > 1)
		temp_own = sigma;
	else
		temp_own =sigma/temp_own;

	double S0 = 0;
	for(j=0;j<count[0];j++)
	{
#ifdef RBF
		S0 += alpha[j]*rbffun(-s_euclid[j],temp_own);
#else
		S0 += alpha[j]*atanfun(s_euclid[j],temp_own);
#endif
	}
		



//CLASSS 2
	distance_euclid = model->spread_euclid[1];
	for(j=count[0];j<l;j++)
	{
		s_euclid[j] = distance1(x_test,x[j]);  
		distance_euclid += s_euclid[j];    //sum of linear distances of x to class 0
#ifdef RBF
		tempS[1] += alpha[j]*rbffun(-s_euclid[j],sigma);
#else
		tempS[1] += alpha[j]*atanfun(s_euclid[j],sigma);
#endif
	}

	temp_own = (distance_euclid*count[0]*(count[0]-1))/(count[1]*(count[1]+1)*model->spread_euclid[0]);
        if(temp_own > 1)
		temp_own = sigma;
	else
		temp_own =sigma/temp_own;

	double S1 = 0;
	for(j=count[0];j<l;j++)
	{
#ifdef RBF
		S1 += alpha[j]*rbffun(-s_euclid[j],temp_own);
#else
		S1 += alpha[j]*atanfun(s_euclid[j],temp_own);
#endif

	}

	
//	S0 = (S0 - tempS[1]);
//	S1 = (S1 - tempS[0]);
	S0 = (S0 - tempS[1] + max(count[0]+1,count[1]) + 0.5)/(2*max(count[0]+1,count[1])+1);
	S1 = (S1 - tempS[0] + max(count[0],count[1]+1) + 0.5)/(2*max(count[0],count[1]+1)+1);
	printf("%.10e %.10e\n",S0,S1);
	
	return !(S0 > S1);
}

double predict_prob(struct imbl_model *model,const struct vector_node *x_test)
{
    int j;
    struct vector_node **x = model->x;
    double sigma = model->sigma;
    double *alpha = model->alpha;
    int *count = model->count;	
	double tempS [2] = {0,0};

	int l = model->l;
	double *s_euclid = Malloc(double,l);   //to keep the linear distances of class 1, assuming x is in class 1, for later use

	for(j=0;j<count[0];j++)
	{
                s_euclid[j] = distance1(x_test,x[j]);
		tempS[0] += alpha[j]*rbffun(-s_euclid[j],sigma);
	}	


	double distance_euclid = 0;
	
	for(j=count[0];j<l;j++)
	{
		s_euclid[j] = distance1(x_test,x[j]);  
		distance_euclid += s_euclid[j];    //sum of linear distances of x to class 0
#ifdef RBF
			tempS[1] += alpha[j]*rbffun(-s_euclid[j],sigma);
#else
			tempS[1] += alpha[j]*atanfun(s_euclid[j],sigma);
#endif
	}

//	double S0 = (tempS[0] - tempS[1] + count[1] + 0.5)/(l+1);   //pred confidence assuming x is in class 0
//	double S0 = (tempS[0] - tempS[1] + max(count[0]+1,count[1]) + 0.5)/(2*max(count[0]+1,count[1])+1);   //pred confidence assuming x is in class 0
	double S0 = tempS[0] - tempS[1];   //pred confidence assuming x is in class 0

	double temp_own = (((count[1]+1)*count[1])*model->spread_euclid[0])/((count[0]*(count[0]-1))*(model->spread_euclid[1] + distance_euclid));
	double S1 = 0;
	for(j=count[0];j<l;j++)
	{
#ifdef RBF
		S1 += alpha[j]*rbffun(-s_euclid[j],temp_own);
#else
		S1 += alpha[j]*atanfun(s_euclid[j],temp_own);
#endif

	}
//	S1 = (S1 - tempS[0] + count[0] + 0.5)/(l+1);
//	S1 = (S1 - tempS[0] + max(count[0],count[1]+1) + 0.5)/(2*max(count[0],count[1]+1)+1);
	S1 = S1 - tempS[0];   //pred confidence assuming x is in class 0
	printf("%.10e %.10e\n",S0,S1);
	
	return S0;//!(S0 > S1);
}

