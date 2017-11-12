
#include "l1tf.h"

#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericVector l1tf(Rcpp::NumericVector y, double lambda, bool debug=false) {
    int n = y.size();
    Rcpp::NumericVector out(n);

    // int l1tf(const int n, const double *y, const double lambda, double *x);
    l1tf_impl(n, REAL(y), lambda, REAL(out), debug);

    return out;
}


// [[Rcpp::export]]
double l1tf_lambdamax(Rcpp::NumericVector y, bool debug) {
    
    // double l1tf_lambdamax(const int n, double *y);
    double max = l1tf_lambdamax_impl(y.size(), REAL(y), debug);

    return max;
}
