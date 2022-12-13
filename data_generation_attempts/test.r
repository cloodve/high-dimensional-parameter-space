
likelihood <- function(d, p){
  l <- 1
  for (i in d){
    if (i == 1){
      l <- l*p
    } else {
      l <- l *(1-p)
    }
  }
  return(l)
}

calc_max_likelihood_params <- function(flips, number_params){
  samples <- split(flips, ceiling(seq_along(flips)/(length(flips)/number_params)))
  p_vals <-c()
  for (i in 1 :length(samples)){
    p <- sum(samples[[i]])/length(samples[[i]])
    p_vals <- c(p_vals, p)
  }
  return(p_vals)
}

calc_likelihood <- function(flips, params){
  samples <- split(flips, ceiling(seq_along(flips)/(length(flips)/length(params))))
  likelihoods <-c()
  for (i in 1 :length(samples)){
    l <- likelihood(samples[[i]], params[i])
    likelihoods <- c(likelihoods, l)
  }
  return(prod(likelihoods))
}

calc_LR <- function(d, ps1, ps2){
  l1 <- calc_likelihood(d, ps1)
  l2 <- calc_likelihood(d, ps2)
  lr <- l1/l2
  lrt <- -2 * (log(lr))
  return(lrt)
}

calc_LRTs <- function(flips, num_params_model_1, num_params_model_2){
  LRTs <- c()
  for (j in 1: length(flips)){
    f <- flips[[j]]
    max_probs_model_1 <- calc_max_likelihood_params(f, num_params_model_1)
    max_probs_model_2 <- calc_max_likelihood_params(f, num_params_model_2)
    statistic <- calc_LR(f, max_probs_model_1, max_probs_model_2)
    LRTs <- append (LRTs, statistic)
  }    
  return(LRTs)
}






flipcoin <- function(p,n){
  f <- sample(c(0,1), n, replace= TRUE, prob = c(1-p,p))
  return (f)
}

# Flip a coin 1000 times 1000 times
flips_per_experiment = 1000
n_experiments = 3
prob_heads = c(.6)
flips <- list()
for (i in 1:n_experiments){
  flips = c(flips, list((flipcoin(prob_heads,flips_per_experiment))))
}



LRTs <- calc_LRTs(flips, 1, 2)