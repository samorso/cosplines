# R functions
# Splines data generating process
# (assume same b-spline basis for both dimensions)
#' @export
r_factor_splines_2d <- function(C,M,n,d){
  z <- runif(n)
  eps <- matrix(runif(n*d),nc=d)
  P <- predict(M,z)
  obs <- matrix(nr=n,nc=d)
  for(i in seq_len(d)){
    P2 <- predict(M,eps[,i])
    obs[,i] <- splines_new_obs(C,P,P2)
  }
  return(obs)
}

# Objective function
#' @export
of_smm <- function(C,M,n,d,q,m_hat,B){
  m_tilde <- matrix(nr=length(m_hat),nc=B)
  for(i in seq_len(B)){
    set.seed(2141+i)
    x <- r_factor_splines_2d(C,M,n,d)
    m_tilde[,i] <- average_moments(x,q)
  }
  of <- norm(m_hat-rowMeans(m_tilde),type="2")
  return(of)
}

# Objective function (with positivity constraint)
#' @export
of_smm2 <- function(C,M,n,d,q,m_hat,B){
  C2 <- exp(C)
  m_tilde <- matrix(nr=length(m_hat),nc=B)
  for(i in seq_len(B)){
    set.seed(2141+i)
    x <- r_factor_splines_2d(C2,M,n,d)
    m_tilde[,i] <- average_moments(x,q)
  }
  of <- norm(m_hat-rowMeans(m_tilde),type="2")
  return(of)
}

# Objective function (with (0,1) constraint)
#' @export
of_smm3 <- function(C,M,n,d,q,m_hat,B){
  C2 <- boot::inv.logit(C)
  m_tilde <- matrix(nr=length(m_hat),nc=B)
  for(i in seq_len(B)){
    set.seed(2141+i)
    x <- r_factor_splines_2d(C2,M,n,d)
    m_tilde[,i] <- average_moments(x,q)
  }
  of <- norm(m_hat-rowMeans(m_tilde),type="2")
  return(of)
}
