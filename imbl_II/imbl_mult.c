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
	int i,j,cind;
	struct imbl_model *model = (struct imbl_model *) malloc(sizeof(struct imbl_model));


	int l = problem->l;

	int nr_class = problem->nr_class;
	struct vector_node ** x;
	double *K = Malloc(double,l*l);
	double *U = Malloc(double,l*l);
	double *B = Malloc(double,l);
	int *start = Malloc(int,nr_class+1);
	int *count = problem->count;
	double * spread_euclid = Malloc(double,nr_class);
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

    int Nm = max_val(count,nr_class);
	int Nmax = 2*Nm;
	double b = Nm + 0.5;

	model->Nm = Nm;	
	model->x = x;
	model->start = start;
	model->count = count;
	model->l = l;
	model->sigma = sigma;
	model->nr_class = nr_class;


	for(cind=0;cind<nr_class;cind++)
	{
		spread_euclid[cind] = 0;
		for(i=start[cind];i<start[cind+1];i++)
		{
			B[i] = b;
			K(i,i) = Nmax;
			for(j=i+1;j<start[cind+1];j++)
			{
#ifdef NORM_EUCLID
				double temp = distance2(x[j],x[i],variances[cind]);
#else
				double temp = distance1(x[j],x[i]);
#endif
				spread_euclid[cind] += temp;
				U(i,j) = temp;
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

	int max_spread_ind = 0;
	double max_spread = 0;
	double *spread_euclid_correct = Malloc(double,nr_class);
	for(cind=0;cind<nr_class;cind++)
	{
		spread_euclid_correct[cind] = spread_euclid[cind]/(count[cind]*(count[cind]-1));
		if(spread_euclid_correct[cind] > max_spread)
		{
			max_spread_ind = cind;
			max_spread = spread_euclid_correct[cind];
		}
	}
	
	model->spread_euclid = spread_euclid;	
	model->spread_euclid_correct = spread_euclid_correct;
	model->max_spread_ind = max_spread_ind;
	model->max_spread = max_spread;

	for(i=start[max_spread_ind];i<start[max_spread_ind+1];i++)
	{
		for(j=i+1;j<start[max_spread_ind+1];j++)
		{
#ifdef RBF
			K(i,j) = -rbffun(-U(i,j),sigma);
#else	
			K(i,j) = -atanfun(U(i,j),sigma);
#endif
		}
	}
		
	
	for(cind=0;cind<nr_class;cind++)
	{	
		if(cind == max_spread_ind)
			continue;

		double sigma_c = sigma*max_spread/spread_euclid_correct[cind];
		for(i=start[cind];i<start[cind+1];i++)
		{
			for(j=i+1;j<start[cind+1];j++)
			{
#ifdef RBF
				K(i,j) = -rbffun(-U(i,j),sigma_c);
#else	
				K(i,j) = -atanfun(U(i,j),sigma_c);
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
    int j,cind;
    struct vector_node **x = model->x;
    double sigma = model->sigma;
    double *alpha = model->alpha;
    int *count = model->count;	
	int *start = model->start;
	int nr_class = model->nr_class;
	int l = model->l;
	double *tempS = Malloc(double,nr_class);  //the sum of similarities from x to each class
	double *S = Malloc(double,nr_class);
	double *s_euclid = Malloc(double,l);   //to keep the linear distances of class 1, assuming x is in class 1, for later use
//	double *distance_euclid = Malloc(double,nr_class);  //each class's spread assuming x is in it
//	double distance_euclid;// = Malloc(double,nr_class);  //each class's spread assuming x is in it
//	double *max_spread_test = Malloc(double,nr_class);  //max spread of the problem when assuming x is in some class
//	int *max_spread_ind_test = Malloc(int,nr_class);  //the class with the max spread when assuming x is in some class
	for(cind=0;cind<nr_class;cind++)
	{
//		distance_euclid[cind] = model->spread_euclid[0];
		double distance_euclid = model->spread_euclid[cind];
//		distance_euclid = model->spread_euclid[0];
		tempS[cind] = 0;
		S[cind]= 0;
		for(j=start[cind];j<start[cind+1];j++)
		{
			s_euclid[j] = distance1(x_test,x[j]);  
//			distance_euclid[cind] += s_euclid[j];    //sum of linear distances of x to class 0
			distance_euclid += s_euclid[j];    //sum of linear distances of x to class 0
#ifdef RBF
			tempS[cind] += alpha[j]*rbffun(-s_euclid[j],sigma);
#else
			tempS[cind] += alpha[j]*atanfun(s_euclid[j],sigma);
#endif
		}	
//		distance_euclid[cind] /= (count[cind]*(count[cind]+1));
		distance_euclid /= (count[cind]*(count[cind]+1));
		double sigma_c = sigma;				//the  sigma to be used for cooperations for the class x is assumed to be in
//		if(distance_euclid[cind] > model->max_spread)

		if(distance_euclid >= model->max_spread)
		{
//			sigma_c = sigma;
			
//			max_spread_test[cind] = distance_euclid[cind];
//			max_spread_ind_test[cind] = cind;
		}
		else if(cind != model->max_spread_ind)
		{
//			sigma_c = sigma*model->max_spread/max_spread_test[cind];
//			sigma_c = sigma*model->max_spread/max_spread_test;
			sigma_c = sigma*model->max_spread/distance_euclid;
//			max_spread_test[cind] = model->max_spread;
//			max_spread_ind_test[cind] = model->max_sprear_ind;
		}
		else
		{
			double temp = distance_euclid;
//			double temp = distance_euclid[cind];
//			max_spread_ind_test[cind] = cind;
			for(j=0;j<nr_class;j++)
			{
				if(j == cind)
					continue;
				if(model->spread_euclid_correct[j] > temp)
				{
					temp = model->spread_euclid_correct[j];
//					max_spread_ind_test[cind] = j;
				}
			}
//			max_spread_test[cind] = temp;
//			sigma_c = sigma*model->spread_euclid_correct[j]->max_spread/max_spread_test[cind];
			sigma_c = sigma*temp/distance_euclid;//model->spread_euclid_correct[j]/->max_spread/max_spread_test[cind];
		}
		for(j=start[cind];j<start[cind+1];j++)
		{
#ifdef RBF
			S[cind] += alpha[j]*rbffun(-s_euclid[j],sigma_c);
#else
			S[cind] += alpha[j]*atanfun(s_euclid[j],sigma_c);
#endif
		}
		for(j=0;j<cind;j++)
		{
			S[cind] -= tempS[j];
			S[j] -= tempS[cind];
		}
	}
	double max_S = -INF;
	int max_class = 0;
	for(cind=0;cind<nr_class;cind++)
	{
		S[cind] = (S[cind] + max(count[cind]+1,model->Nm) + 0.5)/(2*max(count[cind]+1,model->Nm)+1);
		if(S[cind] > max_S)
		{
			max_S = S[cind];
			max_class = cind;
		}
//		printf("%.10e ",S[cind]);
	}
//	printf("\n");
//	return !(S0 > S1);
	return max_class;
}

double predict_prob(struct imbl_model *model,const struct vector_node *x_test)
{
/*    int j;
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
*/
	return 0;
}

