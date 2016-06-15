/*
 * =====================================================================================
 * 
 *        Filename:  ibl.h
 * 
 *     Description:  The C header file for all the procedures and variables used by ibl-train.c, ibl-predict.c and ibl.c.
 *  		     This version is different than others in that, the code hasbeen cleaned, functions that we think will not be used any more have been removed, rather than just commented.  
 *		     Also, starting from this version, we will try to keep a better track of changes and revisions.
 *         Version:  1.0
 *         Created:  02/23/2007 02:15:12 PM MST
 *        Revision:  1.0.0
 *        Compiler:  gcc
 * 
 *          Author:  Karen Hovsepian
 *         Company:  New Mexico Tech
 *           Email:  karen@nmt.edu
 * 
 * =====================================================================================
 */

#define MAX_NUM_ELEMENTS 100000000
#define K(I,J)		K[I+(J)*l]
#define U(I,J)		U[I+(J)*l]

#define INF HUGE_VAL
#define M_PI       3.14159265358979323846

#define MAX_NR_CLASS 40

#define INF_BOUND		1e+30
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct vector_node
{
	int index;
	double value;
};

struct learning_problem
{
	int l;
	int *label;
	struct vector_node **x;

	int *count;
	int *y;
	int nr_class;
	int dim;
};

struct imbl_model
{
	int l;
	struct vector_node **x;
	double *alpha;
	double sigma;
	int nr_class;
	int *start;
	int *count;
	double *spread_euclid;
	double *r;
};

struct imbl_model * train(struct learning_problem *prob,double sigma);
int predict(struct imbl_model *model,const struct vector_node *x);
double predict_prob(struct imbl_model *model,const struct vector_node *x);

double kernel1(const struct vector_node *x, const struct vector_node *y, double sigma);
double kernel2(const struct vector_node *x, const struct vector_node *y, double sigma,double *spreads);

double distance1(const struct vector_node *x, const struct vector_node *y);
double distance2(const struct vector_node *x, const struct vector_node *y,double *spreads);

void cross_validation(const struct learning_problem *prob, double sigma, int nr_fold, double *target);
void leave_one_out(const struct learning_problem *prob, double sigma, double *target);
double * get_variances (struct vector_node **x,int start,int l,int numfeatures);	
int max_val(int *vals,int l);
double max_valdbl(double *vals,int l);
