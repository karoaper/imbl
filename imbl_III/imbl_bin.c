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
	int i,j;

	struct imbl_model *model = (struct imbl_model *) malloc(sizeof(struct imbl_model));
	int l = problem->l;
	model->l = l;
	struct vector_node ** x;

	double *K = Malloc(double,l*l);
	double *B = Malloc(double,l);

	model->sigma = sigma;
	int start [2];
	model->count_class_one = problem->count[0];
	int *count = problem->count;
	model->count = count;
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

	model->x = x;


#ifdef NORM_EUCLID
	double * variances_overall = get_variances(x,0,l,problem->dim);	
	double * variances_c1 = get_variances(x,0,count[0],problem->dim);	
	double * variances_c2 = get_variances(x,count[0],count[1],problem->dim);	
#endif
	double averagesims [3] = {0,0,0};
	int Nm = max(count[0],count[1]);
	int Nmax = 2*Nm;

	double b = Nm + 0.5;
	for(i=0;i<model->count_class_one;i++)
	{
		B[i] = b;
		for(j=i+1;j<model->count_class_one;j++)
		{
#ifdef NORM_EUCLID
			double temp = kernel2(x[j],x[i],sigma,variances_c1);
#else
			double temp = kernel1(x[j],x[i],sigma);
#endif
			K(i,j) = -temp;
			averagesims[0] += temp;
		}
		
		for(j=model->count_class_one;j<l;j++)
		{
#ifdef NORM_EUCLID
			double temp = kernel2(x[j],x[i],sigma,variances_overall);
#else
			double temp = kernel1(x[j],x[i],sigma);
#endif
			K(i,j) = temp;
			averagesims[2] += temp;
		}
	}
	for(i=model->count_class_one;i<l;i++)
	{
		B[i] = b;
		for(j=i+1;j<l;j++)
		{
#ifdef NORM_EUCLID
			double temp = kernel2(x[j],x[i],sigma,variances_c2);
#else
			double temp = kernel1(x[j],x[i],sigma);
#endif
			K(i,j) = -temp;
			averagesims[1] += temp;
		}
	}

	double r1,r2;
	r1 = Nmax+1+(2*averagesims[0]-averagesims[2])/count[0];
	r2 = Nmax+1+(2*averagesims[1]-averagesims[2])/count[1];	      	

	for(i=0;i<model->count_class_one;i++)
		K(i,i) = r1;

	for(i=model->count_class_one;i<l;i++)
		K(i,i) = r2;
	
	model->r1 = 2*max(count[0]+1,count[1])+1+(2*averagesims[0]-averagesims[2])/(count[0]+1);
	model->r2 = 2*max(count[0],count[1]+1)+1+(2*averagesims[1]-averagesims[2])/(count[1]+1);

	dpotrf('U', l,K, l, &i);
	dpotrs ('U', l, 1, K, l,B, l, &i);
	model->alpha = B;
	free(K);
	free(perm);
	return model;
}

int predict(struct imbl_model *model,const struct vector_node *x_test)
{

	double S = 0;
	int j;
	struct vector_node **x = model->x;
	double sigma = model->sigma;
	double *alpha = model->alpha;
	double sumtoclasses [2]= {0,0};	
	for(j=0;j<model->count_class_one;j++)
	{
		double temp = kernel1(x_test,x[j],sigma);
		S += temp*alpha[j];
		sumtoclasses[0] += temp;

	}
	for(j=model->count_class_one;j<model->l;j++)
	{
		double temp = kernel1(x_test,x[j],sigma);
		S -= temp*alpha[j];
		sumtoclasses[1] += temp;
	}

	double Stemp = (S + max(model->count[0]+1,model->count[1])+0.5)/(model->r1 + (2*sumtoclasses[0]-sumtoclasses[1])/(model->count[0]+1));
	double Stemp1 = (-S + max(model->count[0],model->count[1]+1)+0.5)/(model->r2 + (2*sumtoclasses[1]-sumtoclasses[0])/(model->count[1]+1));

	printf("%.10e %.10e %.10e\n",S,Stemp,Stemp1);
	return !(Stemp > Stemp1);
}

double predict_prob(struct imbl_model *model,const struct vector_node *x_test)
{

	double S = 0;
	int j;
	struct vector_node **x = model->x;
	double sigma = model->sigma;
	double *alpha = model->alpha;
	double sumtoclasses [2]= {0,0};	
	for(j=0;j<model->count_class_one;j++)
	{
		double temp = kernel1(x_test,x[j],sigma);
		S += temp*alpha[j];
		sumtoclasses[0] += temp;

	}
	for(j=model->count_class_one;j<model->l;j++)
	{
		double temp = kernel1(x_test,x[j],sigma);
		S -= temp*alpha[j];
		sumtoclasses[1] += temp;
	}

	double Stemp = (S + model->count[1]+0.5)/(model->r1 + (2*sumtoclasses[0]-sumtoclasses[1])/(model->count[0]+1));
	double Stemp1 = (-S + model->count[0]+0.5)/(model->r2 + (2*sumtoclasses[1]-sumtoclasses[0])/(model->count[1]+1));

	printf("%.10e %.10e %.10e\n",S,Stemp,Stemp1);
	return Stemp;// > Stemp1);
}
