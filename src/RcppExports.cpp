// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// l1tf
arma::colvec l1tf(arma::colvec yvec, double lambda, bool debug);
RcppExport SEXP _RcppL1TF_l1tf(SEXP yvecSEXP, SEXP lambdaSEXP, SEXP debugSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type yvec(yvecSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< bool >::type debug(debugSEXP);
    rcpp_result_gen = Rcpp::wrap(l1tf(yvec, lambda, debug));
    return rcpp_result_gen;
END_RCPP
}
// l1tf_lambdamax
double l1tf_lambdamax(arma::colvec yvec, bool debug);
RcppExport SEXP _RcppL1TF_l1tf_lambdamax(SEXP yvecSEXP, SEXP debugSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type yvec(yvecSEXP);
    Rcpp::traits::input_parameter< bool >::type debug(debugSEXP);
    rcpp_result_gen = Rcpp::wrap(l1tf_lambdamax(yvec, debug));
    return rcpp_result_gen;
END_RCPP
}
// superlu_lambdamax
double superlu_lambdamax(const arma::vec& y);
RcppExport SEXP _RcppL1TF_superlu_lambdamax(SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(superlu_lambdamax(y));
    return rcpp_result_gen;
END_RCPP
}
// superlu_l1tf
arma::vec superlu_l1tf(const arma::vec y_vec, const double lambda);
RcppExport SEXP _RcppL1TF_superlu_l1tf(SEXP y_vecSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec >::type y_vec(y_vecSEXP);
    Rcpp::traits::input_parameter< const double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(superlu_l1tf(y_vec, lambda));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RcppL1TF_l1tf", (DL_FUNC) &_RcppL1TF_l1tf, 3},
    {"_RcppL1TF_l1tf_lambdamax", (DL_FUNC) &_RcppL1TF_l1tf_lambdamax, 2},
    {"_RcppL1TF_superlu_lambdamax", (DL_FUNC) &_RcppL1TF_superlu_lambdamax, 1},
    {"_RcppL1TF_superlu_l1tf", (DL_FUNC) &_RcppL1TF_superlu_l1tf, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_RcppL1TF(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
