// imbl_simple.cpp : Defines the entry point for the console application.
//
#include <stdlib.h>
#include <string.h>
#include "imbl.h"
#include <math.h>
#include <stdio.h>
#include <ctype.h>
#include <sys/times.h>

//double (*kernel[2])(const struct vector_node *, const struct vector_node *, double) = {&rbf,&rbf_sqrt};

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define EMPTY_CLASS -100000000
//#define DEFAULT_MIN_CARR 0.00000001

void parse_command_line(int argc, char **argv);
void read_problem_sparse(const char *filename);
void do_cross_validation();
void do_leave_one_out();

//int chosenkernel = RBF;
//int solving_method = LAPACK_CHOL_REG;

void test(FILE *input,char *test_file);

double sigma;
struct learning_problem prob;		// set by read_problem
struct imbl_model *model;
struct vector_node *x_space;
int cross_validation_val = 0;
int loo_val = 0;
int nr_fold;
char * test_file;
char * input_file_name;
int do_prob;

int main(int argc, char ** argv)
{
	read_problem_sparse(argv[1]);
	parse_command_line(argc, (char **) argv);

	if(cross_validation_val)
		do_cross_validation();
	if(loo_val)
		do_leave_one_out();
	if(!cross_validation_val && !loo_val)
	{
		model = train(&prob,sigma);

		FILE *testfile = fopen(test_file,"r");
		test(testfile,test_file);
		fclose(testfile);
	}
	return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double *target = Malloc(double,prob.l);
	struct tms start, end;
//	for(i=0;i<20;i++)
//	{
		times(&start);
		cross_validation(&prob,sigma,nr_fold,target);
		times(&end);
		printf("%ld ",end.tms_utime - start.tms_utime);
//	}
	if(do_prob)
	{
		for(i=0;i<prob.l;i++)
			printf("%d %.40e\n",prob.y[i],target[i]);
	}
	else
	{
		for(i=0;i<prob.l;i++)
		{
			if(target[i] != -1)
			{
				if(target[i] == prob.y[i])
					++total_correct;
			}
		}
//		printf("%g %d\n",100.0*total_correct/prob.l,end.tms_utime-start.tms_utime);
		printf("%g\n",100.0*total_correct/prob.l);
	}
	free(target);
}


void do_leave_one_out()
{
	int i;
	int total_correct = 0;
	double *target = Malloc(double,prob.l);
	leave_one_out(&prob,sigma,target);
	
	for(i=0;i<prob.l;i++)
	{
		if(target[i] != EMPTY_CLASS)
		{
			if(target[i] == prob.y[i])
				++total_correct;
		}
	}	
	printf("%g\n",100.0*total_correct/prob.l);
	free(target);
}


void parse_command_line(int argc, char **argv)
{
	int i;
	for(i=2;i<argc;i++)
	{
		switch(argv[i][1])
		{
			case 's':
				sigma = atof(argv[++i]);
				break;
			case 'L':
				loo_val = 1;
				break;
			case 'v':
				cross_validation_val = 1;
				nr_fold = atoi(argv[++i]);
				break;
			case 't':
				test_file = argv[++i];
				break;
/*			case 'k':
				chosenkernel = atoi(argv[++i]);
				break;
*/
			case 'b':
				do_prob = 1;
				break;
//			case 'm':
//				solving_method = atoi(argv[++i]);
//				break;
			default:
				fprintf(stderr,"unknown option\n");
		}
	}	
}


void read_problem_sparse(const char *filename)
{
	int elements, max_index;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				++prob.l;
				// fall through,
				// count the '-1' element
			case ':':
				++elements;
				break;
			case EOF:
				goto out;
			default:
				;
		}
	}
out:
	rewind(fp);

	prob.x = Malloc(struct vector_node *,prob.l);
	prob.y = Malloc(int,prob.l);
	x_space = Malloc(struct vector_node,elements);

	prob.count = Malloc(int,MAX_NR_CLASS);
	prob.label = Malloc(int,MAX_NR_CLASS);

	int nr_class = 0;
	max_index = 0;
	int j=0;
	int i;
	for(i=0;i<prob.l;i++)
	{
		int label;
		prob.x[i] = &x_space[j];
		fscanf(fp,"%d",&label);

		//begin define count, label and y

		int k;

		for(k=0;k<nr_class;k++)
		{
			if(label == prob.label[k])
			{
				++prob.count[k];
				break;
			}
		}
		prob.y[i] = k;
		if(k == nr_class)
		{
			prob.label[nr_class] = label;
			prob.count[nr_class] = 1;
			++nr_class;
		}

		//end define count, label and y
	
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
			++j;
		}	
out2:
		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
	}
	prob.dim = max_index;
	prob.nr_class = nr_class;
	fclose(fp);
}

void test(FILE *input,char *test_file)
{
	int correct = 0;
    	int total = 0;
	int max_nr_attr = 64;

	struct vector_node *x = (struct vector_node *) malloc(max_nr_attr*sizeof(struct vector_node));

//    int nr_class=2;
                        
    while(1)
    {
                    
        int c;
        int target, v;
        int i=0;
        if (fscanf(input,"%d",&target)==EOF)
			break;
        

        while(1)
        {
                if(i>=max_nr_attr-1)    // need one more for index = -1
                {
                        max_nr_attr *= 2;
                        x = (struct vector_node *) realloc(x,max_nr_attr*sizeof(struct vector_node));
                }

                do {
                        c = getc(input);
                        if(c=='\n' || c==EOF) goto out2;
                } while(isspace(c));
                ungetc(c,input);
                fscanf(input,"%d:%lf",&x[i].index,&x[i].value);
                ++i;
        }

out2:
		x[i++].index = -1;
//		printf("%d ",target);
		if(do_prob)
			printf("%d %.40e\n", target,predict_prob(model,x));
		else
		{
			v = predict(model,x);
			if(prob.label[v] == target)
				++correct;
			++total;
		}
	}

	if(!do_prob)
		printf("p %lf\n",(double)correct/total*100);
}

double kernel1(const struct vector_node *x, const struct vector_node *y, double sigma)
{
	double sum = 0;
	while(x->index != -1 && y->index !=-1)
	{
		if(x->index == y->index)
		{
			double d = x->value - y->value;
#ifdef MAHALA
			sum += fabs(d);
#else
			sum += d*d;
#endif
			++x;
			++y;
		}
		else
		{
			if(x->index > y->index)
			{	
#ifdef MAHALA
				sum += y->value;
#else
				sum += y->value*y->value;
#endif
				++y;
			}
			else
			{
#ifdef MAHALA
				sum += x->value;
#else
				sum += x->value*x->value;
#endif
				++x;
			}
		}
	}

	while(x->index != -1)
	{
#ifdef MAHALA
		sum += x->value;
#else
		sum += x->value*x->value;
#endif
		++x;
	}
	while(y->index != -1)
	{
#ifdef MAHALA
		sum += y->value;
#else
		sum += y->value*y->value;
#endif
		++y;
	}
#ifdef RBF
	#ifdef SQRT
		return exp(-sqrt(sum)*sigma);
	#else
		return exp(-sum*sigma);
	#endif
#else
	#ifdef SQRT
		return 1-2/M_PI*atan(sqrt(sum)*sigma);
	#else
		return 1-2/M_PI*atan(sum*sigma);
	#endif
#endif
}

double kernel2(const struct vector_node *x, const struct vector_node *y, double sigma,double *spreads)
{
	double sum = 0;
	int i =0;
	while(x->index != -1 && y->index !=-1)
	{
		if(x->index == y->index)
		{
			double d = x->value - y->value;
			sum += (d*d)/spreads[i++];
			++x;
			++y;
		}
		else
		{
			if(x->index > y->index)
			{	
				sum += (y->value * y->value)/spreads[i++];
				++y;
			}
			else
			{
				sum += (x->value * x->value)/spreads[i++];
				++x;
			}
		}
	}

	while(x->index != -1)
	{
		sum += (x->value * x->value)/spreads[i++];
		++x;
	}
	while(y->index != -1)
	{
		sum += (y->value * y->value)/spreads[i++];
		++y;
	}

#ifdef RBF
	#ifdef SQRT
		return exp(-sqrt(sum)*sigma);
	#else
		return exp(-sum*sigma);
	#endif
#else
	#ifdef SQRT
		return 1-2/M_PI*atan(sqrt(sum)*sigma);
	#else
		return 1-2/M_PI*atan(sum*sigma);
	#endif
#endif
}



double distance1(const struct vector_node *x, const struct vector_node *y)
{
	double sum = 0;
	while(x->index != -1 && y->index !=-1)
	{
		if(x->index == y->index)
		{
			double d = x->value - y->value;
#ifdef MAHALA
			sum += fabs(d);
#else
			sum += d*d;
#endif
			++x;
			++y;
		}
		else
		{
			if(x->index > y->index)
			{	
#ifdef MAHALA
				sum += y->value;
#else
				sum += y->value*y->value;
#endif
				++y;
			}
			else
			{
#ifdef MAHALA
				sum += x->value;
#else
				sum += x->value*x->value;
#endif
				++x;
			}
		}
	}

	while(x->index != -1)
	{
#ifdef MAHALA
		sum += x->value;
#else
		sum += x->value*x->value;
#endif
		++x;
	}
	while(y->index != -1)
	{
#ifdef MAHALA
		sum += y->value;
#else
		sum += y->value*y->value;
#endif
		++y;
	}

#ifdef SQRT
	return sqrt(sum);
#else
	return sum;
#endif
}

double distance2(const struct vector_node *x, const struct vector_node *y, double *spreads)
{
	double sum = 0;
	int i =0;
	while(x->index != -1 && y->index !=-1)
	{
		if(x->index == y->index)
		{
			double d = x->value - y->value;
			sum += (d*d)/spreads[i++];
			++x;
			++y;
		}
		else
		{
			if(x->index > y->index)
			{	
				sum += (y->value * y->value)/spreads[i++];
				++y;
			}
			else
			{
				sum += (x->value * x->value)/spreads[i++];
				++x;
			}
		}
	}

	while(x->index != -1)
	{
		sum += (x->value * x->value)/spreads[i++];
		++x;
	}
	while(y->index != -1)
	{
		sum += (y->value * y->value)/spreads[i++];
		++y;
	}

#ifdef SQRT
	return sqrt(sum);
#else
	return sum;
#endif
}





double * get_variances (struct vector_node **x,int start,int l,int numfeatures)
{
	double *means = Malloc(double,numfeatures);
	double *variances = Malloc(double,numfeatures);
        struct vector_node *x_temp;
	int i;

	for(i=0;i<numfeatures;i++)
	{
		means[i] =0;
		variances[i] = 0;
	}
	for(i=start;i<start+l;i++)
	{
		x_temp = x[i];   
		while(x_temp->index != -1)
                {
			means[x_temp->index-1] += x_temp->value;
			++x_temp;
		}
	}
	for(i=0;i<numfeatures;i++)
	{
		means[i] /= l;
	}
	for(i=start;i<start+l;i++)
	{
		x_temp = x[i];   
		while(x_temp->index != -1)
                {
			double temp = x_temp->value - means[x_temp->index-1];
			variances[x_temp->index-1] += temp*temp;
			++x_temp;
		}
	}
	for(i=0;i<numfeatures;i++)
	{
		variances[i] /= (l-1);
	}
	return variances;
}

void leave_one_out(const struct learning_problem *prob, double sigma, double *target)
{
	int l = prob->l;
	int i,j;
	for(i=0;i<l;i++)
	{
		struct learning_problem subprob;

		subprob.l = l-1;
		subprob.nr_class = prob->nr_class;

		subprob.x = Malloc(struct vector_node*,subprob.l);
		subprob.y = Malloc(int,subprob.l);
		subprob.count = prob->count;
		subprob.label = prob->label;		
		subprob.count[prob->y[i]]--;
		subprob.dim = prob->dim;
		if(i>0)
			subprob.count[prob->y[i-1]]++;
		int k=0;
		for(j=0;j<l;j++)
		{	
			if(j != i)
			{
				subprob.x[k] = prob->x[j];
				subprob.y[k] = prob->y[j];
				++k;
			}
		}
		struct imbl_model *submodel = train(&subprob,sigma);
		target[i] = predict(submodel,prob->x[i]);
	}
}


void cross_validation(const struct learning_problem *prob, double sigma, int nr_fold, double *target)
{
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int i,j;
	int nr_class = prob->nr_class;

	int *start = Malloc(int, nr_class);
	int *perm = Malloc(int, l);

    int * count = prob->count;

	//------begin group classes


	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[prob->y[i]]] = i;
		++start[prob->y[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

//---------end group classes


	
	
	int *fold_count = Malloc(int,nr_fold);
	int *index = Malloc(int,l);

	int **fold_class_count = Malloc(int *,nr_fold);

	//uncomment line below if you want a different randomization each new time
//	srand( (unsigned)time( NULL ) );

	for(i=0;i<l;i++)
		index[i]=perm[i];
	int c;
	for (c=0; c<nr_class; c++)
		for(i=0;i<count[c];i++)
		{
			int j = i+rand()%(count[c]-i);
			int temp = index[start[c]+j];
			index[start[c]+j] = index[start[c]+i];
			index[start[c]+i] = temp;
		}

	for(i=0;i<nr_fold;i++)
	{
		fold_count[i] = 0;
		fold_class_count[i] = Malloc(int,nr_class);
		for (c=0; c<nr_class;c++)
		{
			int temp_count = (i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
			fold_class_count[i][c] = count[c]-temp_count;
			fold_count[i]+=temp_count;

		}

	}
	
	
	fold_start[0]=0;
	for (i=1;i<=nr_fold;i++)
		fold_start[i] = fold_start[i-1]+fold_count[i-1];
	
	for (c=0; c<nr_class;c++)
	{
		for(i=0;i<nr_fold;i++)
		{
			int begin = start[c]+i*count[c]/nr_fold;
			int end = start[c]+(i+1)*count[c]/nr_fold;
			for(j=begin;j<end;j++)
			{
				perm[fold_start[i]] = index[j];
				fold_start[i]++;
			}
		}
	}
	fold_start[0]=0;
	for (i=1;i<=nr_fold;i++)
		fold_start[i] = fold_start[i-1]+fold_count[i-1];
	free(start);
	free(count);
	free(index);
	free(fold_count);

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int k=0;

		struct learning_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct vector_node*,subprob.l);

		subprob.y = Malloc(int,subprob.l);
		subprob.count = fold_class_count [i];
		
		subprob.label = prob->label;
		subprob.nr_class = nr_class;

		subprob.dim = prob->dim;
		

		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];

			subprob.y[k] = prob->y[perm[j]];
			++k;
		}

		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];

			subprob.y[k] = prob->y[perm[j]];
			++k;
		}

		struct imbl_model *submodel = train(&subprob,sigma);
		if(do_prob)
		{
			for(j=begin;j<end;j++)
			{
				target[perm[j]] = predict_prob(submodel,prob->x[perm[j]]);
			}
		}
		else
		{
			for(j=begin;j<end;j++)
			{
				target[perm[j]] = predict(submodel,prob->x[perm[j]]);
			}
		}
		free(submodel);
	}
	free(fold_start);
	free(perm);
}


double max_valdbl(double *vals,int l)
{
	double temp = -INF;
	int i;
	for(i=0;i<l;i++)
	{
		if(vals[i] > temp)
			temp = vals[i];
	}
	return temp;
}

int max_val(int *vals,int l)
{
	int temp = (int)-INF;
	int i;
	for(i=0;i<l;i++)
	{
		if(vals[i] > temp)
			temp = vals[i];
	}
	return temp;
}

