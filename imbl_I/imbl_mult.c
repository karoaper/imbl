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
#include "acml.h"

//extern int solving_method;
//extern int chosenkernel;
//extern double (*kernel[2])(const struct vector_node *, const struct vector_node *, double);

#define min(a,b)	(a < b ? a : b)
#define max(a,b)	(a < b ? b : a)


struct imbl_model * train(struct learning_problem *problem, double sigma)
{
	int i,j,cind;
	
	struct imbl_model *model = (struct imbl_model *) malloc(sizeof(struct imbl_model));
	int l = problem->l;
	struct vector_node ** x;

	double *K = Malloc(double,l*l);
	double *B = Malloc(double,l);
	int nr_class = problem->nr_class;


	int *start = Malloc(int,nr_class+1);
	int *perm = Malloc(int,l);
	int *count = problem->count;
	//------begin group classes

	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	for(i=0;i<l;i++)
	{
		perm[start[problem->y[i]]] = i;
		++start[problem->y[i]];
	}

	start[0] = 0;
	for(i=1;i<=nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	//---------end group classes

	x = Malloc(struct vector_node *,l);
	
	for(i=0;i<l;i++)
	{
		x[i] = problem->x[perm[i]];
	}

	model->x = x;
	model->start = start;
	model->count = count;
	model->sigma = sigma;
	model->nr_class = nr_class;
	model->l = l;


#ifdef NORM_EUCLID
	double ** variances = Malloc(double *,nr_class);
	double * variances_overall = get_variances(x,0,l,problem->dim);

	for(i=0;i<nr_class;i++)
		variances[i] = get_variances(x,start[i],count[i],problem->dim);
#endif

	int Nm = max_val(problem->count,nr_class);
	int Nmax = 2*Nm;
	double b = Nm + 0.5;
	for(cind=0;cind<nr_class;cind++)
	{
		for(i=start[cind];i<start[cind+1];i++)
		{
			B[i] = b;
			K(i,i) = Nmax;
			for(j=i+1;j<start[cind+1];j++)
			{
#ifdef NORM_EUCLID
				K(i,j) = -kernel2(x[j],x[i],sigma,variances[cind]);
#else
				K(i,j) = -kernel1(x[j],x[i],sigma);
#endif
			}
			for(j=start[cind+1];j<l;j++)
			{
#ifdef NORM_EUCLID
				K(i,j) = kernel2(x[j],x[i],sigma,variances_overall);
#else
				K(i,j) = kernel1(x[j],x[i],sigma);
#endif
			}
		}
	}

	

	dpotrf('U', l,K, l, &i);
    	dpotrs ('U', l, 1, K, l,B, l, &i);
	model->alpha = B;
	free(K);
	free(perm);
	return model;
}

int predict(struct imbl_model *model,const struct vector_node *x_test)
{
	int j,cind;
	struct vector_node **x = model->x;
	double sigma = model->sigma;
	double *alpha = model->alpha;
	double maxval = -INF;
	int maxint = 0;	
//	int l = model->l;
	int *start = model->start;
	int nr_class = model->nr_class;
	double *Stemp = Malloc(double,nr_class);

	for(cind=0;cind<nr_class;cind++)
	{
		Stemp[cind] =0;
		for(j=start[cind];j<start[cind+1];j++)
		{
			Stemp[cind] += kernel1(x_test,x[j],sigma)*alpha[j];
		}
	}
	for(cind=0;cind<nr_class;cind++)
	{
		double S  = Stemp[cind];
		for(j=0;j<cind;j++)
		{
			S -= Stemp[j];
		}
		for(j=cind+1;j<nr_class;j++)
		{
			S -= Stemp[j];
		}
		if(S > maxval)
		{
			maxval = S;
			maxint = cind;
		}
//		printf("%.10e ",S);
	}
//	printf("\n");
//	printf("%.10e %.10e\n",S,(S + max(model->count[0]+1,model->count[1]) + 0.5)/(2*max(model->count[0]+1,model->count[1]) + 1));
	return maxint;
}

double predict_prob(struct imbl_model *model,const struct vector_node *x_test)
{
/*	double S = 0;//model->Nm + 0.5;
	int j;
	struct vector_node **x = model->x;
	double sigma = model->sigma;
	double *alpha = model->alpha;
		for(j=0;j<start[cind];j++)
		{
			S += kernel1(x_test,x[j],sigma)*alpha[j];
		}
		for(j=model->count_class_one;j<model->l;j++)
		{
			S -= kernel1(x_test,x[j],sigma)*alpha[j];
		}
//	return (S + model->l- model->count_class_one+ 0.5)/(model->l+1);
	return (S + max(model->count[0]+1,model->count[1]) + 0.5)/(2*max(model->count[0]+1,model->count[1]) + 1);
*/
	return 0;
}
