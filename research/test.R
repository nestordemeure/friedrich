
library(SDMTools)

# training data
train_x <- c(0.8, 1.2, 3.8, 4.2, 7)
train_y <- c(3  , 4  , -2 , -2, -2)
prior <- function(x) 0
prior <- function(x) 3*cos(x)

# the kernel used
squared_exponential <- function(x,y, length_scale=1)
{
   distance <- sqrt((x - y)**2)
   kernel <- exp((distance / length_scale)**2 / -2)
   return(kernel)
}

# std but computed relativ to a given value and not to the mean
excentred_sd <- function(data, center)
{
   var <- sum((data - center)**2) / (length(data) - 1)
   return(sqrt(var))
}

# predicted value
# with or without prior
k_mean <- function(x, use_prior=FALSE)
{
   weights <- squared_exponential(x, train_x)

   if(use_prior)
   {      
      prior_weight <- (1 - max(weights)) / (1 + sum(weights))
      target_y <- train_y - prior(train_x) # residual trick
      y_prior <- c(0, target_y) # prior is now 0
      weights_prior <- c(prior_weight, weights)
      y <- wt.mean(y_prior, weights_prior) + prior(x) # adding prior back
   }
   else 
   {
      y <- wt.mean(train_y, weights)
   }

   return(y)
}

# predicted sd
k_sd <- function(x, use_prior=FALSE)
{
   weights <- squared_exponential(x, train_x)
   prior_weight <- (1 - max(weights)) / (1 + sum(weights))
   if(use_prior)
   {
      train_y <- train_y - prior(train_x) # residual trick, prior is now 0
      
      prior_y <- c(0, train_y)
      prior_weights <- c(prior_weight, weights)
      y_predicted <- wt.mean(prior_y, prior_weights)
   }
   else 
   {
      y_predicted <- wt.mean(train_y, weights)
   }
   
   # sd without additional information
   s <- excentred_sd(train_y, y_predicted)
   # weighted sd
   y_prior <- c(y_predicted + s, y_predicted - s, train_y)
   weights_prior <- c(prior_weight/2, prior_weight/2, weights)
   std <- wt.sd(y_prior, weights_prior)
   
   return(std)
}

#------------------------------------------------------------------------------

library(ggplot2)

sd(train_y)

x <- c(-500:1000) / 100
y <- sapply(x, function(x) k_mean(x,TRUE))
std <- sapply(x, function(x) k_sd(x,TRUE) )
y_plus <- y + std
y_minus <- y - std

X11()
ggplot() +
geom_point(aes(x=train_x, y=train_y), color='red') +
geom_line(aes(x=x, y=y), color='blue') +
geom_line(aes(x=x, y=y_plus), color='grey') +
geom_line(aes(x=x, y=y_minus), color='grey') #+
#geom_line(aes(x=x, y=prior(x)), color='green', linetype='dashed')
   
X11()
ggplot() +
   geom_point(aes(x=train_x, y=0), color='red') +
   geom_line(aes(x=x, y=0), color='blue') +
   geom_line(aes(x=x, y=std), color='grey') +
   geom_line(aes(x=x, y=-std), color='grey')
