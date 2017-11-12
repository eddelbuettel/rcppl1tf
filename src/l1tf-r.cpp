cd 
#include "l1tf.h"

#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector l1tf_R(Rcpp::NumericVector y, double lambda, bool debug) {
    int n = y.size();
    Rcpp::NumericVector out(n);

    // int l1tf(const int n, const double *y, const double lambda, double *x);
    l1tf(n, REAL(y), lambda, REAL(out), debug);

    return out;
}


// [[Rcpp::export]]
double l1tf_lambdamax_R(Rcpp::NumericVector y, bool debug) {
    
    // double l1tf_lambdamax(const int n, double *y);
    double max = l1tf_lambdamax(y.size(), REAL(y), debug);

    return max;
}
