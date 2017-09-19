// [[Rcpp::depends(RcppArmadillo,BH)]]
#include <RcppArmadillo.h>
#include <random>
#include <math.h>
#include <numeric>
#include <boost/math/distributions/gamma.hpp>

// [[Rcpp::interfaces(r, cpp)]]

// ---------------------
// Sample functions
// ---------------------
arma::vec vec_empirical_cdf(
  arma::vec& x
){
  unsigned int n(x.n_elem);
  arma::uvec i(n);
  arma::uvec j(n);
  arma::vec y(n);

  i = arma::sort_index(x);
  j = arma::sort_index(i);
  j += 1;
  y = arma::conv_to<arma::vec>::from(j);
  n += 1;
  y /= n;

  return y;
}

arma::mat mat_empirical_cdf(
  arma::mat& x
){
  unsigned int n(x.n_rows);
  unsigned int D(x.n_cols);
  arma::mat y(n,D);
  arma::vec p1(n);

  for(unsigned int d(0); d < D; ++d){
    p1 = x.col(d);
    y.col(d) = vec_empirical_cdf(p1);
  }

  return y;
}

//' @export
// [[Rcpp::export]]
double spearman_rank(
  arma::mat& x // empirical CDFs
){
  unsigned int n(x.n_rows);
  arma::vec y(n);
  double rho;

  y = x.col(0);
  y %= x.col(1);

  rho = arma::sum(y) / n * 0.12e2 - 0.3e1;
  return rho;
}

double quantile_dependence_low(
  arma::mat& x, // empirical CDFs
  double q
){
  unsigned int n(x.n_rows);
  arma::mat m1(n,2);
  arma::vec p1(n);
  double lambda;

  m1 = x;
  m1.transform([&q](double val) { return (val <= q) ? double(0.1e1) : double(0.0); } );
  p1 = m1.col(0);
  p1 %= m1.col(1);

  lambda = arma::sum(p1) / q / n;

  return lambda;
}

double quantile_dependence_high(
  arma::mat& x, // empirical CDFs
  double q
){
  unsigned int n(x.n_rows);
  arma::mat m1(n,2);
  arma::vec p1(n);
  double t1;
  double lambda;

  m1 = x;
  m1.transform([&q](double val) { return (val > q) ? double(0.1e1) : double(0.0); } );
  p1 = m1.col(0);
  p1 %= m1.col(1);
  t1 = 0.1e1 - q;
  t1 *= n;

  lambda = arma::sum(p1) / t1;

  return lambda;
}

//' @export
// [[Rcpp::export]]
double quantile_dependence(
  arma::mat& x, // empirical CDFs
  double q
){
  double lambda;

  if(q <= 0.5){
    lambda = quantile_dependence_low(x,q);
  }else{
    lambda = quantile_dependence_high(x,q);
  }

  return lambda;
}

// vector of moments
// [[Rcpp::export]]
arma::vec vec_moments(
    arma::mat& x,
    arma::vec& q
){
  unsigned int p(q.n_elem);
  p += 1;
  arma::vec v(p);

  v(0) = spearman_rank(x);

  for(unsigned int i(1); i<p; ++i){
    v(i) = quantile_dependence(x,q(i-1));
  }

  return v;
}

// Average of estimated moments
//' @export
// [[Rcpp::export]]
arma::vec average_moments(
    arma::mat& x,
    arma::vec& q
){
  unsigned int n(x.n_rows);
  unsigned int D(x.n_cols);
  unsigned int p(q.n_elem);
  p += 1;
  unsigned int d;
  d = D - 1;
  d *= D;
  d /= 2;
  arma::vec v(p, arma::fill::zeros);
  arma::mat m1(n,2);

  for(unsigned int i(0); i < D-1; ++i){
    for(unsigned int j = i+1; j < D; ++j){
      m1.col(0) = x.col(i);
      m1.col(1) = x.col(j);
      v += vec_moments(m1,q);
    }
  }

  v /= d;

  return v;
}

// -----------------
// Parametric functions
// -----------------
//' @export
// [[Rcpp::export]]
arma::mat clayton(
  arma::vec& z,
  arma::mat& eps,
  double alpha
){
  unsigned int n(z.n_elem);
  unsigned int D(eps.n_cols);
  arma::mat x1(n,D);
  arma::mat x2(n,D);
  arma::vec y(n);
  boost::math::gamma_distribution<> dist(alpha,1);

  for(unsigned int i(0);i < n;++i){
    y(i) = boost::math::quantile(dist,z(i));
  }

  x1 = -arma::log(eps);
  x1.each_col() /= y;
  x1 += 0.1e1;
  x2 = arma::pow(x1,-alpha);

  return x2;
}


// -----------------
// Data generating process
// -----------------
//' @export
// [[Rcpp::export]]
arma::vec splines_new_obs(
  arma::vec& coefs,
  arma::mat& B1,
  arma::mat& B2
){
  unsigned int n(B1.n_rows);
  unsigned int p(coefs.n_elem);
  arma::vec x(n);
  arma::rowvec v(p);

  for(unsigned int i(0);i<n;++i){
    v = arma::kron(B1.row(i),B2.row(i));
    x(i) = arma::dot(coefs,arma::trans(v));
  }

  return x;
}

// arma::vec r_factor_splines(
//   arma::mat& C
//   arma::mat& B
// )


// -----------------
// Simulated Method of Moments
// -----------------

