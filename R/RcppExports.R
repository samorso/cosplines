# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' @export
spearman_rank <- function(x) {
    .Call('_cosplines_spearman_rank', PACKAGE = 'cosplines', x)
}

#' @export
quantile_dependence <- function(x, q) {
    .Call('_cosplines_quantile_dependence', PACKAGE = 'cosplines', x, q)
}

vec_moments <- function(x, q) {
    .Call('_cosplines_vec_moments', PACKAGE = 'cosplines', x, q)
}

#' @export
average_moments <- function(x, q) {
    .Call('_cosplines_average_moments', PACKAGE = 'cosplines', x, q)
}

#' @export
clayton <- function(z, eps, alpha) {
    .Call('_cosplines_clayton', PACKAGE = 'cosplines', z, eps, alpha)
}

#' @export
splines_new_obs <- function(coefs, B1, B2) {
    .Call('_cosplines_splines_new_obs', PACKAGE = 'cosplines', coefs, B1, B2)
}

# Register entry points for exported C++ functions
methods::setLoadAction(function(ns) {
    .Call('_cosplines_RcppExport_registerCCallable', PACKAGE = 'cosplines')
})
