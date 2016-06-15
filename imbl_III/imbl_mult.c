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
	int i,j,k,cind;

	struct imbl_model *model = (struct imbl_model *) malloc(sizeof(struct imbl_model));
	int l = problem->l;
	int nr_class = problem->nr_class;

	struct vector_node ** x;

	double *K = Malloc(double,l*l);
	double *B = Malloc(double,l);

	int *start = Malloc(int,nr_class+1);
	int *count = problem->count;
	int *perm = Malloc(int,l);

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
	


#ifdef NORM_EUCLID
	double ** variances = Malloc(double *,nr_class);
	double * variances_overall = get_variances(x,0,l,problem->dim);

	for(i=0;i<nr_class;i++)
		variances[i] = get_variances(x,start[i],count[i],problem->dim);
#endif

//	double * averagesims_coop  = Malloc(double,nr_class);
	double * averagesims_comp  = Malloc(double,nr_class);
	for(i=0;i<nr_class;i++)
	{
//		averagesims_coop[i]  = 0;
		averagesims_comp[i]  = 0;
	}
//	double *r = Malloc(double,nr_class);

	int Nm = max_val(count,nr_class);
	int Nmax = 2*Nm;
	double b = Nm + 0.5;

	model->r = Malloc(double,nr_class);
	model->x = x;
	model->start = start;
	model->count = count;
	model->sigma = sigma;
	model->l = l;
	model->nr_class = nr_class;
	model->Nm = Nm;	

	for(cind=0;cind<nr_class;cind++)
	{	
		double averagesims_coop = 0;
		for(i=start[cind];i<start[cind+1];i++)
		{
			B[i] = b;
			for(j=i+1;j<start[cind+1];j++)
			{
#ifdef NORM_EUCLID
				double temp = kernel2(x[j],x[i],sigma,variances[cind]);
#else
				double temp = kernel1(x[j],x[i],sigma);
#endif
				K(i,j) = -temp;
//				averagesims_coop[cind] += temp;
				averagesims_coop += temp;
			}
			for (k=cind+1;k<nr_class;k++)
			{
				for(j=start[k];j<start[k+1];j++)
				{
#ifdef NORM_EUCLID
					double temp = kernel2(x[j],x[i],sigma,variances_overall);
#else
					double temp = kernel1(x[j],x[i],sigma);
#endif
					K(i,j) = temp;
					averagesims_comp[cind] += temp;
					averagesims_comp[k] += temp;
				}
			}
		}
		double r = Nmax+1+(2*averagesims_coop-averagesims_comp[cind])/count[cind];
		model->r[cind] = 2*max(count[cind]+1,Nm)+1+(2*averagesims_coop-averagesims_comp[cind])/(count[cind]+1);
		for(i=start[cind];i<start[cind+1];i++)
		{
			K(i,i) = r;
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
    int *count = model->count;	
	int *start = model->start;
	int nr_class = model->nr_class;
//	int l = model->l;
	double *tempS = Malloc(double,nr_class);  //the sum of similarities from x to each class
	double * sumtoclasses = Malloc(double,nr_class);	

	for(cind=0;cind<nr_class;cind++)
	{
		sumtoclasses[cind] = 0;
		tempS[cind] = 0;
		for(j=start[cind];j<start[cind+1];j++)
		{
			double temp = kernel1(x_test,x[j],sigma);
			tempS[cind] += temp*alpha[j];
			sumtoclasses[cind] += temp;
		}
	}
	int max_class= 0;
	double max_S = -INF;
	for(cind=0;cind<nr_class;cind++)
	{
		double S = tempS[cind];
		double temp = 0;
		for(j=0;j<cind;j++)
		{
			S -= tempS[j];
			temp += sumtoclasses[j];
		}
		for(j=cind+1;j<nr_class;j++)
		{
			S -= tempS[j];
			temp += sumtoclasses[j];
		}
		S = (S + max(count[cind]+1,model->Nm)+0.5)/(model->r[cind] + (2*sumtoclasses[cind]-temp)/(model->count[cind]+1));
		if(S > max_S)
		{
			max_class = cind;
			max_S = S;
		}
//		printf("%.10e ",S);
	}
//	printf("\n");
	return max_class;
}

double predict_prob(struct imbl_model *model,const struct vector_node *x_test)
{

/*	double S = 0;
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
*/
	return 0;
}
