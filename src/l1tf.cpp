// l2tf.cpp -- based on l1tf.c by Koh, Kim and Boyd and also GPL'ed
//
// also see    https://github.com/eddelbuettel/l1tf
//             https://github.com/hadley/l1tf
//             https://github.com/thanos-mad-titan/l1tf

/* l1tf.c
 *
 * Copyright (C) 2007 Kwangmoo Koh, Seung-Jean Kim and Stephen Boyd.
 *
 * See the file "COPYING.TXT" for copyright and warranty information.
 *
 * Author: Kwangmoo Koh (deneb1@stanford.edu)
 */

// this define overcomes an old design wart where (one col or row) vectors are
// always returned as a (one col or row) matrix -- now the col vectors collapse
// does not seem to work here though -- to be seen
// #define RCPP_ARMADILLO_RETURN_COLVEC_AS_VECTOR 1

// Use SuperLU for sparse matrix
#define ARMA_USE_SUPERLU 1

#include <RcppArmadillo.h>

#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

#ifndef F77_CALL
#define F77_CALL(x) x ## _
#endif

/* macro functions */
//#define  max(x,y)       ((x)>(y)?(x):(y))
//#define  min(x,y)       ((x)<(y)?(x):(y))

#define  TRUE           (1)
#define  FALSE          (0)

/* constant variables for fortran call */
const double done       = 1.0;
const double dtwo       = 2.0;
const double dminusone  =-1.0;
const int    ione       = 1;
const int    itwo       = 2;
const int    ithree     = 3;
const int    iseven     = 7;

/* function declaration */
void   Dx(const int n, const double *x, double *y); /* y = D*x */
void   DTx(const int n, const double *x, double *y); /* y = D'*x */
void   yainvx(int n, const double a, const double *x, double *y); /* y = a./x */
//void   vecset(int n, const double val, double *dst);
double vecsum(int n, const double *src);

/****************************************************************************
 *                                                                          *
 *              l1tf : main routine for l1 trend filtering                  *
 *                                                                          *
 ****************************************************************************/
// [[Rcpp::export]]
arma::colvec l1tf(arma::colvec yvec, double lambda, bool debug=false) {

    int n = yvec.size();
    arma::colvec xvec(n);
    double *x = &(xvec[0]);
    double *y = &(yvec[0]);

    /* parameters */
    const double ALPHA      = 0.01; /* linesearch parameter (0,0.5] */
    const double BETA       = 0.5;  /* linesearch parameter (0,1) */
    const double MU         = 2;    /* IPM parameter: t update */
    const double MAXITER    = 40;   /* IPM parameter: max iter. of IPM */
    const double MAXLSITER  = 20;   /* IPM parameter: max iter. of linesearch */
    const double TOL        = 1e-4; /* IPM parameter: tolerance */

    /* dimension */
    const int    m          = n-2;  /* length of Dx */

    /* matrix of size 3xm */
    double *S, *DDTF;
    arma::mat Smat(3, m); S = &(Smat[0]);
    arma::mat DDTFmat(7, m); DDTF = &(DDTFmat[0]);

    /* vector of size n */
    double *DTz;
    arma::vec DTzvec(n); DTz = &(DTzvec[0]);

    /* vector of size m */
    double *z;                      /* dual variable */
    arma::vec Z = arma::ones(m); z = &(Z[0]);
    double *mu1, *mu2;              /* dual of dual variables */
    arma::vec MU1 = arma::zeros(m); mu1 = &(MU1[0]);
    arma::vec MU2 = arma::zeros(m); mu2 = &(MU2[0]);
    double *f1, *f2;                /* constraints */
    arma::vec F1 = arma::ones(m) * (-lambda); f1 = &(F1[0]);
    arma::vec F2 = arma::ones(m) * (-lambda); f2 = &(F2[0]);

    double *dz, *dmu1, *dmu2;       /* search directions */
    arma::vec DZ(m); dz = &(DZ[0]);
    arma::vec DMU1(m); dmu1 = &(DMU1[0]);
    arma::vec DMU2(m); dmu2 = &(DMU2[0]);
    double *w, *rz, *tmp_m2, *tmp_m1, *Dy, *DDTz;
    arma::vec W(m); w = &(W[0]);
    arma::vec RZ(m); rz = &(RZ[0]);
    arma::vec TMPM1(m); tmp_m1 = &(TMPM1[0]);
    arma::vec TMPM2(m); tmp_m2 = &(TMPM2[0]);
    arma::vec Dyvec(m); Dy = &(Dyvec[0]);
    arma::vec DDTzvec(m); DDTz = &(DDTzvec[0]);
    double norm2_res, norm2_newres;

    double pobj1, pobj2, pobj, dobj, gap;
    double t;                       /* barrier update parameter */
    double step, ratio;
    double *dptr;

    int iters, lsiters;             /* IPM and linesearch iterations */
    int i, info;
    int *IPIV;
    arma::ivec IPIVvec(m); IPIV = &(IPIVvec[0]);
    int ddtf_chol;

    /* memory allocation */
    //S       = malloc(sizeof(double)*m*3);
    //DDTF    = malloc(sizeof(double)*m*7);

    //DTz     = malloc(sizeof(double)*n);
    //Dy      = malloc(sizeof(double)*m);
    //DDTz    = malloc(sizeof(double)*m);

    //z       = malloc(sizeof(double)*m);
    //mu1     = malloc(sizeof(double)*m);
    //mu2     = malloc(sizeof(double)*m);
    //f1      = malloc(sizeof(double)*m);
    //f2      = malloc(sizeof(double)*m);
    //dz      = malloc(sizeof(double)*m);
    //dmu1    = malloc(sizeof(double)*m);
    //dmu2    = malloc(sizeof(double)*m);
    //w       = malloc(sizeof(double)*m);
    //rz      = malloc(sizeof(double)*m);
    //tmp_m1  = malloc(sizeof(double)*m);
    //tmp_m2  = malloc(sizeof(double)*m);

    //IPIV    = malloc(sizeof(int)*m);

    /* INITIALIZATION */

    /* DDT : packed representation of symmetric banded matrix
     *
     * fortran style storage (should be transposed in C)
     *  6  6  6  ...  6  6  6
     * -4 -4 -4  ... -4 -4  *
     *  1  1  1  ...  1  *  *
     */

    ddtf_chol = TRUE;
    /* DDTF stores Cholesky factor of DDT in packed symm.-band representation */
    dptr = DDTF;
    for (i = 0; i < m; i++) {
        *dptr++ = 6.0;
        *dptr++ =-4.0;
        *dptr++ = 1.0;
    }
    F77_CALL(dpbtrf)("L",&m,&itwo,DDTF,&ithree,&info);
    if (info > 0) {  /* if Cholesky factorization fails, try LU factorization */
        if (debug) Rprintf("Changing to LU factorization\n");
        ddtf_chol = FALSE;
        dptr = DDTF;
        for (i = 0; i < m; i++) {
            dptr++;
            dptr++;
            *dptr++ = 1.0;
            *dptr++ =-4.0;
            *dptr++ = 6.0;
            *dptr++ =-4.0;
            *dptr++ = 1.0;
        }
        F77_CALL(dgbtrf)(&m,&m,&itwo,&itwo,DDTF,&iseven,IPIV,&info);
    }

    Dx(n,y,Dy);

    /* variable initialization */
    t       = -1;
    step    =  1;

    //vecset(m,0.0,z);
    //vecset(m,1.0,mu1);
    //vecset(m,1.0,mu2);
    //vecset(m,-lambda,f1);
    //vecset(m,-lambda,f2);

    DTx(m,z,DTz); /* DTz = D'*z */
    Dx(n,DTz,DDTz); /* DDTz = D*D'*z */

    if (debug) Rprintf("%s %13s %12s %8s\n","Iteration","Primal obj.", \
            "Dual obj.","Gap");

    /*---------------------------------------------------------------------*
     *                          MAIN LOOP                                  *
     *---------------------------------------------------------------------*/
    for (iters = 0; iters <= MAXITER; iters++) {
        double zTDDTz;

        /* COMPUTE DUALITY GAP */
        zTDDTz = F77_CALL(ddot)(&n,DTz,&ione,DTz,&ione);

        F77_CALL(dcopy)(&m,Dy,&ione,w,&ione); /* w = D*y-(mu1-mu2)  */
        F77_CALL(daxpy)(&m,&dminusone,mu1,&ione,w,&ione);
        F77_CALL(daxpy)(&m,&done,mu2,&ione,w,&ione);

        F77_CALL(dcopy)(&m,Dy,&ione,tmp_m2,&ione); /* tmp_m2 = D*y-D*D'*z */
        F77_CALL(daxpy)(&m,&dminusone,DDTz,&ione,tmp_m2,&ione);

        F77_CALL(dcopy)(&m,w,&ione,tmp_m1,&ione); /* tmp_m1 = w*(D*D')^-1*w */
        if (ddtf_chol == TRUE) {
            F77_CALL(dpbtrs)("L",&m,&itwo,&ione,DDTF,&ithree,tmp_m1,&m,&info);
        } else {
            F77_CALL(dgbtrs)("N",&m,&itwo,&itwo,&ione,DDTF,&iseven,IPIV,tmp_m1,&m,&info);
        }

        pobj1 = 0.5*F77_CALL(ddot)(&m,w,&ione,tmp_m1,&ione)
                +lambda*(vecsum(m,mu1)+vecsum(m,mu2));
        pobj2 = 0.5*zTDDTz + lambda*F77_CALL(dasum)(&m,tmp_m2,&ione);
        pobj  = std::min(pobj1,pobj2);
        dobj  =-0.5*zTDDTz+F77_CALL(ddot)(&m,Dy,&ione,z,&ione);

        gap   = pobj - dobj;

        if (debug) Rprintf("%6d %15.4e %13.5e %10.2e\n",iters,pobj,dobj,gap);

        /* STOPPING CRITERION */

        if (gap <= TOL) {
            if (debug) Rprintf("Solved\n");
            F77_CALL(dcopy)(&n,y,&ione,x,&ione);
            F77_CALL(daxpy)(&n,&dminusone,DTz,&ione,x,&ione);
            return xvec;
        }

        if (step >= 0.2) {
            t = std::max(2*m*MU/gap, 1.2*t);
        }

        /* CALCULATE NEWTON STEP */

        F77_CALL(dcopy)(&m,DDTz,&ione,rz,&ione); /* rz = D*D'*z-w */
        F77_CALL(daxpy)(&m,&dminusone,w,&ione,rz,&ione);

        yainvx(m,+1.0/t,f1,dz); /* dz = r = D*y-D*D'*z+(1/t)./f1-(1/t)./f2 */
        yainvx(m,-1.0/t,f2,tmp_m1);
        F77_CALL(daxpy)(&m,&done,tmp_m1,&ione,dz,&ione);
        F77_CALL(daxpy)(&m,&done,tmp_m2,&ione,dz,&ione);

        dptr = S; /* S = D*D'-diag(mu1./f1-mu2./f2) */
        for (i = 0; i < m; i++) {
            *dptr++ = 6-mu1[i]/f1[i]-mu2[i]/f2[i];
            *dptr++ =-4.0;
            *dptr++ = 1.0;
        }

        F77_CALL(dpbsv)("L",&m,&itwo,&ione,S,&ithree,dz,&m,&info); /* dz=S\r */

        norm2_res = F77_CALL(ddot)(&m,rz,&ione,rz,&ione);
        for (i = 0; i < m; i++) {
            double tmp1, tmp2;
            tmp1 = -mu1[i]*f1[i]-(1/t);
            tmp2 = -mu2[i]*f2[i]-(1/t);
            norm2_res += tmp1*tmp1+tmp2*tmp2;

            dmu1[i] = -(mu1[i]+((1/t)+dz[i]*mu1[i])/f1[i]);
            dmu2[i] = -(mu2[i]+((1/t)-dz[i]*mu2[i])/f2[i]);
        }
        norm2_res = sqrt(norm2_res);

        /* BACKTRACKING LINESEARCH */

        ratio = 2;   /* any number larger than 1/0.99 */
        for (i = 0; i < m; i++)
        {
            if (dmu1[i]<0 && -mu1[i]/dmu1[i]<ratio) ratio = -mu1[i]/dmu1[i];
            if (dmu2[i]<0 && -mu2[i]/dmu2[i]<ratio) ratio = -mu2[i]/dmu2[i];
        }
        step = std::min(1.0,0.99*ratio);

        /* compute new values of z, dmu1, dmu2, f1, f2 */
        F77_CALL(daxpy)(&m,&step,dz  ,&ione,z  ,&ione);
        F77_CALL(daxpy)(&m,&step,dmu1,&ione,mu1,&ione);
        F77_CALL(daxpy)(&m,&step,dmu2,&ione,mu2,&ione);

        for (lsiters = 0; lsiters < MAXLSITER; lsiters++) {
            int linesearch_skip;
            double diff_step;

            linesearch_skip = 0;
            for (i = 0; i < m; i++) {
                f1[i] =  z[i]-lambda;
                f2[i] = -z[i]-lambda;
                if (f1[i] > 0 || f2[i] > 0) linesearch_skip = 1;
            }
            if (linesearch_skip != 1) {
                DTx(m,z,DTz); /* rz = D*D'*z-D*y-(mu1-mu2) */
                Dx(n,DTz,DDTz);
                F77_CALL(dcopy)(&m,DDTz,&ione,rz,&ione);
                F77_CALL(daxpy)(&m,&dminusone,Dy,&ione,rz,&ione);
                F77_CALL(daxpy)(&m,&done,mu1,&ione,rz,&ione);
                F77_CALL(daxpy)(&m,&dminusone,mu2,&ione,rz,&ione);

                /* UPDATE RESIDUAL */

                /* compute  norm([rz; -mu1.*f1-1/t; -mu2.*f2-1/t]) */
                norm2_newres = F77_CALL(ddot)(&m,rz,&ione,rz,&ione);
                for (i = 0; i < m; i++) {
                    double tmp1, tmp2;
                    tmp1 = -mu1[i]*f1[i]-(1/t);
                    tmp2 = -mu2[i]*f2[i]-(1/t);
                    norm2_newres += tmp1*tmp1+tmp2*tmp2;
                }
                norm2_newres = sqrt(norm2_newres);

                if (norm2_newres <= (1-ALPHA*step)*norm2_res)
                    break;
            }
            diff_step = -step*(1.0-BETA);
            F77_CALL(daxpy)(&m,&diff_step,dz  ,&ione,z  ,&ione);
            F77_CALL(daxpy)(&m,&diff_step,dmu1,&ione,mu1,&ione);
            F77_CALL(daxpy)(&m,&diff_step,dmu2,&ione,mu2,&ione);
            step *= BETA;
        }
    }
    if (debug) Rprintf("Maxiter exceeded\n");
    F77_CALL(dcopy)(&n,y,&ione,x,&ione);
    F77_CALL(daxpy)(&n,&dminusone,DTz,&ione,x,&ione);
    return xvec;
}

// [[Rcpp::export]]
double l1tf_lambdamax(arma::colvec yvec, bool debug=false) {

    int n = yvec.size();
    double *y = &(yvec[0]);

    int i, m, info;
    double maxval;
    double *vec, *mat, *dptr;
    int *piv;

    m = n-2;
    //vec = malloc(sizeof(double)*m);
    //mat = malloc(sizeof(double)*7*m);
    //piv = malloc(sizeof(int)*m);
    arma::vec VEC(m); vec = &(VEC[0]);
    arma::mat MAT(7,m); mat = &(MAT[0]);
    arma::ivec PIV(m); piv = &(PIV[0]);

    Dx(n,y,vec);
    dptr = mat;
    for (i = 0; i < m; i++) {
        *dptr++ = 6;
        *dptr++ =-4.0;
        *dptr++ = 1.0;
    }

    F77_CALL(dpbsv)("L",&m,&itwo,&ione,mat,&ithree,vec,&m,&info);
    if (info > 0) {  /* if Cholesky factorization fails, try LU factorization */
        if (debug) Rprintf("Changing to LU factorization\n");
        dptr = mat;
        for (i = 0; i < m; i++) {
            dptr++;
            dptr++;
            *dptr++ = 1.0;
            *dptr++ =-4.0;
            *dptr++ = 6.0;
            *dptr++ =-4.0;
            *dptr++ = 1.0;
        }

        F77_CALL(dgbsv)(&m,&itwo,&itwo,&ione,mat,&iseven,piv,vec,&m,&info);
        if (info > 0) return -1.0;  /* if LU fails, return -1 */
    }
    maxval = 0;
    for (i = 0; i < m; i++) {
        if (fabs(vec[i]) > maxval) maxval = fabs(vec[i]);
    }

    return maxval;
}

// [[Rcpp::export]]
double superlu_lambdamax(const arma::vec & y) {
	const int n = y.size();
	const int m = n - 2;  // length of Dx

    arma::mat I2 = arma::eye(m, m);
    arma::mat O2 = arma::zeros(m, 1);
    arma::sp_mat D = arma::sp_mat(arma::join_horiz(I2, arma::join_horiz(O2, O2)) +
                                  arma::join_horiz(O2, arma::join_horiz(-2.0 * I2, O2)) +
                                  arma::join_horiz(O2, arma::join_horiz(O2, I2)));

    arma::sp_mat DDT = D * D.t();
    arma::vec Dy = D * y;
	return arma::norm(arma::spsolve(DDT, Dy), "inf");
}


/* Computes y = D*x, where x has length n
 *
 *     | 1 -2  1  0  0 |
 * y = | 0  1 -2  1  0 |*x
 *     | 0  0  1 -2  1 |
 */
void Dx(const int n, const double *x, double *y) {
    int i;
    for (i = 0; i < n-2; i++,x++)
        *y++ = *x-*(x+1)-*(x+1)+*(x+2); /* y[0..n-3]*/
}

/* Computes y = D^T*x, where x has length n
 *
 *     | 1  0  0 |
 *     |-2  1  0 |
 * y = | 1 -2  1 |*x
 *     | 0  1 -2 |
 *     | 0  0  1 |
 */
void DTx(const int n, const double *x, double *y) {
    int i;
    *y++ = *x;                          /* y[0]     */
    *y++ = -*x-*x+*(x+1);               /* y[1]     */
    for (i = 2; i < n; i++,x++)
        *y++ = *x-*(x+1)-*(x+1)+*(x+2); /* y[2..n-1]*/
    *y++ = *x-*(x+1)-*(x+1); x++;       /* y[n]     */
    *y = *x;                            /* y[n+1]   */
}

/* Computes y = a./x, where x has length n */
void yainvx(int n, const double a, const double *x, double *y) {
    while (n-- != 0)
        *y++ = a/ *x++;
}

// /* Set dst = val, where dst has length n */
// void vecset(int n, const double val, double *dst) {
//     while (n-- != 0)
//         *dst++ = val;
// }

/* Computes sum(x) */
double vecsum(int n, const double *x) {
    double ret = 0.0;
    while (n-- != 0)
        ret += *x++;
    return ret;
}

using namespace arma;

// [[Rcpp::export]]
arma::vec superlu_l1tf(const arma::vec y_vec, const double lambda) {
    const int n = y_vec.size();

	/* parameters */
	const double ALPHA = 0.01; /* linesearch parameter (0,0.5] */
	const double BETA = 0.5;  /* linesearch parameter (0,1) */
	const double MU = 2;    /* IPM parameter: t update */
	const double MAXITER = 40;   /* IPM parameter: max iter. of IPM */
	const double MAXLSITER = 20;   /* IPM parameter: max iter. of linesearch */
	const double TOL = 1e-4; /* IPM parameter: tolerance */

	/* dimension */
	const int    m = n - 2;  /* length of Dx */

	//vec y_vec = vec(y, n);

	mat I2 = eye(m, m);
	mat O2 = zeros(m, 1);
	sp_mat D = sp_mat(join_horiz(I2, join_horiz(O2, O2)) +
                      join_horiz(O2, join_horiz(-2.0 * I2, O2)) +
                      join_horiz(O2, join_horiz(O2, I2)));

	sp_mat DDT = D * D.t();
	mat Dy = D * y_vec;

	mat z = zeros(m, 1);
	mat mu1 = ones(m, 1);
	mat mu2 = ones(m, 1);

	double t = 1e-10;
	double step = std::numeric_limits<double>::infinity();
	double dobj = 0.0;
	unsigned int iter = 0;

	mat f1 = z - lambda;
	mat f2 = -z - lambda;
	mat DTz(n, 1);
	mat DDTz(m, 1);
	mat w(m, 1);
	mat rz(m, 1);
	sp_mat S(m, m);
	mat r(m, 1);
	mat dz(m, 1);
	mat dmu1(m, 1);
	mat dmu2(m, 1);
	mat resDual(m, 1);
	mat newResDual(m, 1);
	mat resCent(2 * m, 1);
	mat newresCent(2 * m, 1);
	mat residual(3 * m, 1);
	mat newResidual(3 * m, 1);
	mat newz(m, 1);
	mat newmu1(m, 1);
	mat newmu2(m, 1);
	mat newf1(m, 1);
	mat newf2(m, 1);
	for (; iter < MAXITER; ++iter) {
		DTz = (z.t() * D).t();
		DDTz = D * DTz;
		w = Dy - (mu1 - mu2);

		// two ways to evaluate primal objective :
		// 1) using dual variable of dual problem
		// 2) using optimality condition
		vec xw = spsolve(DDT, w);

		mat pobj1 = (0.5 * w.t() * (xw)) + lambda * arma::sum(mu1 + mu2);

		mat pobj2 = ((0.5 * DTz.t() * DTz)) + lambda * arma::sum(abs(Dy - DDTz));
		mat pobjm = arma::min(pobj1, pobj2);
		double pobj = pobjm.at(0, 0);
		dobj = std::max((-0.5 * DTz.t() * DTz + Dy.t() * z)[0,0], dobj);
		double gap = pobj - dobj;

		//Stopping criteria
		if (gap <= TOL) {
			//vec x_vec = y_vec - D.t() * z;
			//::memcpy(x, x_vec.memptr(), sizeof(double)* y_vec.n_elem);
			return y_vec - D.t() * z;
		}

		if (step >= 0.2) {
			t = std::max(2.0 * m * MU/gap, 1.2 * t);
		}

		// Calculate Newton Step
		rz = DDTz - w;
		S = DDT - diagmat(mu1/f1 + mu2/f2);
		r = -DDTz + Dy + ((1 / t) / f1) - ((1 / t) / f2);
		dz = mat(spsolve(S, r));
		dmu1 = -(mu1 + ((dz % mu1) + (1 / t)) / f1);
		dmu2 = -(mu2 + ((dz % mu2) + (1 / t)) / f2);

		resDual = rz;
		resCent =  join_vert((-mu1 % f1) - 1 / t, (-mu2 % f2) - 1 / t);
		residual =  join_vert(resDual, resCent);

		// Backtracking linesearch.
		umat  negIdx1 = all(dmu1 < 0.0);
		umat negIdx2 = all(dmu2 < 0.0);
		step = 1.0;

		if (any(vectorise(negIdx1))) {
			step = std::min(step, 0.99*arma::min(-mu1(negIdx1)/dmu1(negIdx1)));
		}

		if (any(vectorise(negIdx2))) {
			step = std::min(step, 0.99*arma::min(-mu2(negIdx2)/ dmu2(negIdx2)));
		}

		for (unsigned int liter = 0; liter < MAXLSITER; ++liter) {
			newz = z + step * dz;
			newmu1 = mu1 + step * dmu1;
			newmu2 = mu2 + step * dmu2;
			newf1 = newz - lambda;
			newf2 = -newz - lambda;

			// Update residual

			//% UPDATE RESIDUAL
			newResDual = DDT * newz - Dy + newmu1 - newmu2;
			newresCent = join_vert((-newmu1 % newf1) - 1 / t, (-newmu2 % newf2) - 1 / t);
			newResidual = join_vert(newResDual, newresCent);

			if ((std::max(arma::max(vectorise(newf1)), arma::max(vectorise(newf2))) < 0.0) && norm(newResidual) <= (1 - ALPHA*step)*norm(residual)) {
				break;
			}

			step = BETA * step;
		}
		z = newz; mu1 = newmu1; mu2 = newmu2; f1 = newf1; f2 = newf2;
	}

	// The solution may be close at this point, but does not meet the stopping
	// criterion(in terms of duality gap).

	if (iter >= MAXITER) {
		//vec x_vec = y_vec - D.t() *z;
		//::memcpy(x, x_vec.memptr(), y_vec.n_elem * sizeof(double));
		return y_vec - D.t() * z;
	}
    return arma::vec();         // not reached
}
