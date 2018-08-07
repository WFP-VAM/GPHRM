#Sys.setenv("DISPLAY"=":0")
#options(menu.graphics=FALSE)

#Load Libraries
library(PrevMap)
library(geoR)
library(doMC)
library(caret)
library(matrixcalc)
library(glmnet)
library(MASS)
library(Matrix)

#fix variance covariance matrix if not Positive SemiDefinite
vcovFixPSD <- function(vcovMtx){
  newMat <- vcovMtx
  iter <- 0
  error <- TRUE
  
  while (error) {
    iter <- iter + 1
    cat("Fixing vcov matrix - iteration ", iter, "\n")
    
    # replace -ve eigen values with small +ve number
    newEig <- eigen(newMat)
    newEig2 <- ifelse(newEig$values < 10e-8, 10e-8, newEig$values)
    # create modified matrix eqn 5 from Brissette et al 2007, inv = transpose for eigenvectors
    newMat <- newEig$vectors %*% diag(newEig2) %*% t(newEig$vectors)
    # normalize modified matrix eqn 6 from Brissette et al 2007
    newMat <- newMat/sqrt(diag(newMat) %*% t(diag(newMat)))
    # force to be symmetric
    newMat <- forceSymmetric(newMat)
    
    # try again
    error <- ifelse(prod(eigen(newMat)$values)<=10e-7,TRUE,FALSE)
  }
  return(newMat)
}
 
#Whiten variables in mean function
whitenSVD <- function(x, tol=0.005){
  
  Nm <- nrow(x)-1
  MEANS <- colMeans(x)
  x.c <- as.matrix(sweep(x,2,MEANS,"-"))
  
  SVD <- svd(x.c, nu = 0)
  SV <- SVD$d 
  if (!is.null(tol)) {
    rank <- sum(SVD$d > (SVD$d[1L] * tol))
    if (rank < ncol(x)) {
      
      SVD$v <- SVD$v[, 1L:rank, drop = FALSE]
      SVD$d <- SVD$d[1L:rank]
    }
  }
  SIGMAs <- SVD$d / sqrt(Nm)
  TRANS <- SVD$v %*% diag(1/SIGMAs)
  RES = x.c %*% TRANS
  
  
  attr(RES, "center") <- MEANS
  attr(RES, "transform") <- TRANS
  attr(RES, "backtransform") <- diag(SIGMAs) %*% t(SVD$v)
  attr(RES, "SingularValues") <- SV
  RES
}

#Whiten & Select Variables in mean function
whitenSlct <- function(Y,X){
  Xzca <- whitenSVD(X)
  
  if(all(Y %in% c(0,1))){
    fam = 'binomial'
  } else if(!is.null(dim(Y))){
    if(dim(Y)[2]==2){
      fam = 'binomial'
      Y <- as.matrix(Y)
    }
  } else {
    fam = 'gaussian'
  }
  cvfit <- cv.glmnet(Xzca,Y,family="binomial",alpha=1,nfolds=5,intercept=FALSE)
  best.lambda <- cvfit$lambda.min
  betas <-coef(cvfit$glmnet.fit,s=best.lambda)
  Xzca <- Xzca[,which(betas!=0)]
  return(Xzca)
}

#fit prevmap
fitPrevGP <- function(dfGPS,dfY,dfX,kappa,method='auto',z='auto',fitTwice='auto'){
  
  colsY <- colnames(dfY)
  #Step 1 - fit linear model and set as prior for betas
  if(is.data.frame(dfX) | is.matrix(dfX)){
    dfXY <- cbind(dfY,dfGPS,dfX)
    fmla <- as.formula(paste('cbind(',colsY[1],',',colsY[2],')~',paste(colnames(dfX),collapse='+',sep=''),sep=''))
  } else {
    dfXY <- cbind(dfY,dfGPS)
    fmla <- as.formula(paste('cbind(',colsY[1],',',colsY[2],')~1',sep=''))
  }
  
  fit.glm <- glm(fmla,data=dfXY,family=binomial)
  
  #Step 2 - fit variogram and set as priors for GP parameters
  vari <- variog(coords = as.matrix(dfGPS),
                 data=log((dfY[1]+0.25)/(dfY[2]+0.25)))
  vari.fit <- variofit(vari, ini.cov.pars = c(2, 0.2),
                       cov.model = "matern",
                       fix.nugget = FALSE, nugget = 0 ,
                       fix.kappa = TRUE, kappa = kappa)
  
  sigmaGP <- ifelse(vari.fit$cov.par[1]>0,vari.fit$cov.par[1],0.01) 
  phiLen <- ifelse(vari.fit$cov.par[2]>0,vari.fit$cov.par[2],0.01)
  nugget <- ifelse(vari.fit$nugget>0,vari.fit$nugget,0.01)
  
  #Step 3 - set priors
  par0 <- c(coef(fit.glm), sigmaGP, phiLen, nugget)
  
  vcovBeta <- vcov(fit.glm)
  if (prod(eigen(vcovBeta)$values)<=10e-7){
    vcovBeta <- vcovFixPSD(vcovBeta)#vcov(fit.glm)
  }
  ctrl.prior <- control.prior(beta.mean = coef(fit.glm), beta.covar = vcovBeta*100,
                              log.normal.sigma2 = c(log(sigmaGP),max(2,2-log10(sigmaGP))), 
                              uniform.phi = c(phiLen/5,phiLen*5),
                              log.normal.nugget = c(log(nugget),max(2,2-log10(nugget))))
                              #floor(log(nugget)-1/(2*nugget^2))
  
  #Step 4 - parse formulae
  dfXY$nN <- rowSums(dfY)
  fmlaGPS <- as.formula(paste('~',paste(colnames(dfGPS),collapse='+')))
  fmla <- as.formula(paste(colnames(dfY)[1],'~',paste(colnames(dfX),collapse='+',sep=''),sep=''))
  
  #Step 5 - choose fitting method and number of iterations if method='auto' and/or z='auto'
  zMC <- z
  zML <- z
  fitTwice <- TRUE
  if(method=='auto' | z=='auto'){
    ctrl.mcmc2  <- control.mcmc.Bayes(10,2,2,
                                      L.S.lim = c(5, 50),
                                      epsilon.S.lim = c(0.025, 0.05), 
                                      start.sigma2 = sigmaGP,start.beta = coef(fit.glm),
                                      start.phi = phiLen, start.nugget = nugget,
                                      start.S = predict(fit.glm))
    sink('/dev/null')
    t0 <- proc.time()
    mdl.Prev <- binomial.logistic.Bayes(fmla,~nN,fmlaGPS,
                                        data=dfXY,kappa=kappa,
                                        control.mcmc=ctrl.mcmc2,control.prior=ctrl.prior)
    t1 <- proc.time()-t0
    sink()
    
    evalsPerHr <- round(3600/(t1[[1]]/10))
    if (evalsPerHr>=4000){
      method <-  'Bayes'
      fitTwice <- TRUE
      zMC <- min(1,round(evalsPerHr/10000,2))
    } else if (evalsPerHr>=2000){
      method <-  'Bayes'
      fitTwice <- FALSE
      zMC <- min(1,round(evalsPerHr/5000,2))
    } else {
      method <- 'MCML'
      fitTwice <- FALSE
      zMC <- min(1,round(evalsPerHr/5000,2))
    }
    zML <- max(1/3,min(1,round(evalsPerHr/5000,2)))
    rm(ctrl.mcmc2,evalsPerHr,t,t0)
  }
  
  #Step 6 - set control parameters
  setIterBurnThin <- function(method,z){
    if(method=='Bayes'){
      iter <- max(2000,round(5000*z))
      burn <- max(1000,round(1000*z))
    } else {
      iter <- max(5000,round(15000*z))
      burn <- max(1000,round(3000*z))
    }
    
    if((iter-burn)%%2>0){burn <- burn-1}
    
    thin <- max(2,round(2*z))
    if(thin>2){
      for(x in rev(seq(2,thin,2))){
        if((iter-burn)%%x==0){
          thin <- x
          break
        }
      }
    }
    return (c('iter'=iter,'burn'=burn,'thin'=thin))
  }
  
  iterBurnThin <- setIterBurnThin('MCML',zML)
  ctrl.mcml <- control.mcmc.MCML(iterBurnThin[1],iterBurnThin[2],iterBurnThin[3])
  
  iterBurnThin <- setIterBurnThin('Bayes',zMC)
  ctrl.mcmc  <- control.mcmc.Bayes(iterBurnThin[1],iterBurnThin[2],iterBurnThin[3],
                                   L.S.lim = c(5, 50),
                                   epsilon.S.lim = c(0.025, 0.05), 
                                   start.sigma2 = sigmaGP,start.beta = coef(fit.glm),
                                   start.phi = phiLen, start.nugget = nugget,
                                   start.S = predict(fit.glm))

  #Step 7 - fit
  if(method=='Bayes'){
    t0 <- proc.time()
    mdl.Prev <- binomial.logistic.Bayes(fmla,~nN,fmlaGPS,
                                        data=dfXY,kappa=kappa,
                                        control.mcmc=ctrl.mcmc,control.prior=ctrl.prior)
    t1 <- proc.time()-t0
    
    #retreive estimates
    sigmaGP <- huber(mdl.Prev$estimate[,dim(mdl.Prev$estimate)[2]-2])$mu
    phiLen <- huber(mdl.Prev$estimate[,dim(mdl.Prev$estimate)[2]-1])$mu
    nugget <- huber(mdl.Prev$estimate[,dim(mdl.Prev$estimate)[2]])$mu
    coefBeta <- apply(mdl.Prev$estimate[,1:(dim(mdl.Prev$estimate)[2]-3)],2,function(x) huber(x)$mu)
    vcovBeta <- cov(mdl.Prev$estimate[,1:(dim(mdl.Prev$estimate)[2]-3)])
    if (prod(eigen(vcovBeta)$values)<=10e-7){
      vcovBeta <- vcovFixPSD(vcovBeta)#vcov(fit.glm)
    }
    
    #fit again if need be
    fitTwice <- ifelse(fitTwice=='auto',ifelse(t1[[1]]<=4000,TRUE,FALSE),fitTwice)
    if(fitTwice){    
      
      #instantiate new priors to fit again
      par0 <- c(coefBeta,sigmaGP, phiLen, nugget)
      ctrl.prior <- control.prior(beta.mean = coefBeta, beta.covar = vcovBeta*10,
                                  log.normal.sigma2 = c(log(sigmaGP),1.5), 
                                  uniform.phi = c(phiLen/1.5,phiLen*1.5),
                                  log.normal.nugget = c(log(nugget),1.5))
      
      ctrl.mcmc  <- control.mcmc.Bayes(iterBurnThin[1],iterBurnThin[2],iterBurnThin[3],
                                       L.S.lim = c(5, 50),
                                       epsilon.S.lim = c(0.025, 0.05), 
                                       start.sigma2 = sigmaGP,start.beta = coefBeta,
                                       start.phi = phiLen, start.nugget = nugget,
                                       start.S = model.matrix(fmla,dfXY) %*% coefBeta)
      
      #fit
      mdl.Prev <- tryCatch({
                    binomial.logistic.Bayes(fmla,~nN,fmlaGPS,
                                            data=dfXY,kappa=kappa,
                                            control.mcmc=ctrl.mcmc,control.prior=ctrl.prior)
                  }, error=function(e){#Try using MCML instead
                    print('Error with MCMC - using MCML instead')
                    binomial.logistic.MCML(fmla,~nN,fmlaGPS,
                                           data=dfXY, kappa=kappa, 
                                           par0=par0,control.mcmc=ctrl.mcml,
                                           start.cov.pars = c(phiLen, nugget/sigmaGP))
                  })
    }
    
  } else { #fit by MCML
    
    t0 <- proc.time()
    mdl.Prev <- tryCatch({
                  binomial.logistic.MCML(fmla,~nN,fmlaGPS,
                                         data=dfXY, kappa=kappa, 
                                         par0=par0,control.mcmc=ctrl.mcml,
                                         start.cov.pars = c(phiLen, nugget/sigmaGP))
                }, error=function(e){
                  print('Error with MCML - using MCMC instead')
                  binomial.logistic.Bayes(fmla,~nN,fmlaGPS,
                                          data=dfXY,kappa=kappa,
                                          control.mcmc=ctrl.mcmc,control.prior=ctrl.prior)
                })
    t1 <- proc.time()-t0
    
    #extract estimates
    if(class(mdl.Prev)=="Bayes.PrevMap"){
      sigmaGP <- huber(mdl.Prev$estimate[,dim(mdl.Prev$estimate)[2]-2])$mu
      phiLen <- huber(mdl.Prev$estimate[,dim(mdl.Prev$estimate)[2]-1])$mu
      nugget <- huber(mdl.Prev$estimate[,dim(mdl.Prev$estimate)[2]])$mu
      coefBeta <- apply(mdl.Prev$estimate[,1:(dim(mdl.Prev$estimate)[2]-3)],2,function(x) huber(x)$mu)
      vcovBeta <- cov(mdl.Prev$estimate[,1:(dim(mdl.Prev$estimate)[2]-3)])
    } else {
      sigmaGP <- exp(mdl.Prev$estimate[(length(mdl.Prev$estimate)-2)])
      phiLen <- exp(mdl.Prev$estimate[(length(mdl.Prev$estimate)-1)])
      nugget <- exp(mdl.Prev$estimate[length(mdl.Prev$estimate)])
      coefBeta <- mdl.Prev$estimate[1:(length(mdl.Prev$estimate)-3)]
      vcovBeta <- mdl.Prev$covariance[1:(dim(mdl.Prev$covariance)[2]-3),1:(dim(mdl.Prev$covariance)[2]-3)]
    }
    if (prod(eigen(vcovBeta)$values)<=10e-7){
      vcovBeta <- vcovFixPSD(vcovBeta)#vcov(fit.glm)
    }

    #take coefficients and fit again if need be
    fitTwice <- ifelse(fitTwice=='auto',ifelse(t1[[1]]<=4000,TRUE,FALSE),fitTwice)
    if(fitTwice){
      
      par0 <- c(coefBeta,sigmaGP, phiLen, nugget)
      ctrl.prior <- control.prior(beta.mean = coefBeta, beta.covar = vcovBeta*10,
                                  log.normal.sigma2 = c(log(sigmaGP),1.5), 
                                  uniform.phi = c(phiLen/1.5,phiLen*1.5),
                                  log.normal.nugget = c(log(nugget),1.5))
      
      ctrl.mcmc  <- control.mcmc.Bayes(iterBurnThin[1],iterBurnThin[2],iterBurnThin[3],
                                       L.S.lim = c(5, 50),
                                       epsilon.S.lim = c(0.025, 0.05), 
                                       start.sigma2 = sigmaGP,start.beta = coefBeta,
                                       start.phi = phiLen, start.nugget = nugget,
                                       start.S = model.matrix(fmla,dfXY) %*% coefBeta)
      
      mdl.Prev <- tryCatch({
                    binomial.logistic.MCML(fmla,~nN,fmlaGPS,
                                           data=dfXY, kappa=kappa, 
                                           par0=par0,control.mcmc=ctrl.mcml,
                                           start.cov.pars = c(phiLen, nugget/sigmaGP))
                  }, error=function(e){
                    print('Error with MCML - using MCMC instead')
                    binomial.logistic.Bayes(fmla,~nN,fmlaGPS,
                                            data=dfXY,kappa=kappa,
                                            control.mcmc=ctrl.mcmc,control.prior=ctrl.prior)
                  })
    }
  }
  
  #edit call information
  mdl.Prev$call[2][[1]] <- fmla
  mdl.Prev$call[4][[1]] <- fmlaGPS
  mdl.Prev$iterBurnThin <- iterBurnThin
  
  return(mdl.Prev)
}

#predict 
prdPrevGP <- function(mdl.Prev,dfGPS,dfX=NULL,xCDThresh=c(0.05,0.1,0.2,0.35,0.5),returnDF=FALSE){
  
  #retrieve optimizer parameters
  iterBurnThin <- mdl.Prev$iterBurnThin
  
  #if model was trained using Markov Chain Monte-Carlo
  if(class(mdl.Prev)=="Bayes.PrevMap"){
    prd.Prev <- spatial.pred.binomial.Bayes(mdl.Prev,
                                            grid=as.matrix(dfGPS),predictors=dfX,
                                            type = "marginal",
                                            scale.predictions = "prevalence",scale.thresholds = "prevalence",
                                            thresholds = xCDThresh, 
                                            quantiles=c(0.05,0.1,0.5,0.9,0.95),
                                            standard.errors = TRUE)
  
  } else {
    # if model was trained using Markov-Chain Maximum Likelihood
    ctrl.mcml <- control.mcmc.MCML(iterBurnThin[1],iterBurnThin[2],iterBurnThin[3])
    prd.Prev <- spatial.pred.binomial.MCML(mdl.Prev,
                                           grid=as.matrix(dfGPS),predictors=dfX,
                                           type = "marginal",control.mcmc = ctrl.mcml,
                                           scale.predictions = "prevalence",scale.thresholds = "prevalence",
                                           thresholds = xCDThresh, 
                                           quantiles=c(0.05,0.1,0.5,0.9,0.95),
                                           standard.errors = TRUE)
  }
  
  #if returning predictions as dataframe - coalesce results from predictions and concatenate into dataframe
  if (returnDF){
    df <- cbind('prevalence'=prd.Prev$prevalence$predictions,
                'stdError'  =prd.Prev$prevalence$standard.errors,
                as.data.frame(prd.Prev$prevalence$quantiles),
                as.data.frame(prd.Prev$exceedance.prob))
    colnames(df)[8:12] <- paste0('x',xCDThresh*100,'%')
    return(df)
  } else {
    return(prd.Prev)  
  }
}

#train/Predict convenience function
trainPredictGP <- function(kappa,trainX,trainY,trainGPS,testX,testGPS,
                           method='auto',z='auto',xCDThresh=c(0.05,0.1,0.2,0.35,0.5),suppress=TRUE){
  
  #suppresses output to screen
  sink.reset <- function(){
    for(i in seq_len(sink.number())){
      sink(NULL)
    }
  }
  sink('/dev/null')
  if(suppress==FALSE){sink()}
  
  #fit model on train data
  mdl.fit <- fitPrevGP(trainGPS,trainY,trainX,kappa,method,z=z)
  #predict on test data
  prd.mdl <- prdPrevGP(mdl.fit,testGPS,testX,xCDThresh=xCDThresh)
  
  #combine results and return as dataframe
  df <- cbind('prevalence'= prd.mdl$prevalence$predictions,
              'stdError'  = prd.mdl$prevalence$standard.errors,
              as.data.frame(prd.mdl$prevalence$quantiles),
              as.data.frame(prd.mdl$exceedance.prob))
  colnames(df)[8:12] <- paste0('x',xCDThresh*100,'%')
  
  #reset suppressed output
  sink.reset()

  return(df)
}

#Evaluate Performance (ignore)
evalPerf <- function(yHat,yAct){
  y <- as.data.frame(cbind('obs'=yAct,'pred'=yHat))
  prf <- defaultSummary(y)
  #prf <- c(prf,'MAPE'=rowMeans(abs((yAct-yHat)/yHat) * 100))
  return(prf)
}

#Find Kappa Matern shape parameter (ignore)
tuneKappa <- function(dfGPS,dfY,dfX){
  
  #Create caret object
  prvMapTune <- list(type="Regression",library="PrevMap",loop=NULL)
  
  #Define tuning parameters
  prvMapTune$parameters <- data.frame(parameter="kappa",class="numeric",label="order")
  
  #Define Training Grid
  prvMapTune$grid <- function(x,y,len=NULL,search="grid") {expand.grid(kappa = seq(0.5, 2.5, 0.25))}
  
  #Define function to fit model
  prvMapTune$fit <- function(x,y,wts,param,lev,last,weights,classProbs, ...){
    dfGPS <- x[,c('Lat','Lng')]
    X <- x[,!(names(x) %in% c('nOK','nFI','Lat','Lng'))]
    Y <- x[,c('nOK','nFI')]
    fitPrevGP(dfGPS,Y,X,kappa=param$kappa,method='Bayes',z=2/5)
  }
  
  #Define function to predict
  prvMapTune$predict <- function(modelFit, newdata, preProc=NULL, submodels=NULL) {
    dfGPS <- newdata[,c('Lat','Lng')]
    newdata <- newdata[,!(names(x) %in% c('nOK','nFI','Lat','Lng'))]
    prd.Prv <- prdPrevGP(modelFit,dfGPS,newdata,z=2/5)
    prd.Prv$prevalence$predictions
  }
  
  #Define function for probabilities
  prvMapTune$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
    predict(modelFit, newdata)
  }
  
  #Define function to sort
  prvMapTune$sort <- function(x) x
  
  #Assemble data into one dataframe
  colnames(dfY) <- c('nOK','nFI')
  colnames(dfGPS) <- c('lat','lng')
  dfY$prevalence <- df$nOK/(df$nOK+df$nFI)
  df <- cbind(dfY,dfGPS,dfX)
  
  # register cores for parallel processing
  registerDoMC(4)
  # train using caret::train()
  set.seed(1984)
  pmTune <- caret::train(y = df[,'prevalence'],
                         x = df[,c(1,2,4,5,c(6:dim(df)[2]))],
                         method = prvMapTune,
                         trControl = trainControl(method='cv',number=5),
                         tuneGrid = expand.grid(kappa = seq(0.5, 2.5, 0.25)))
  
  return(pmTune)
  
}

# #fit using geoCount as an alternative
# library(geoCount)
# library(snowfall)
# library(rlecuyer)  # needed for random number generation when using snowfall
# fitGeoCntGP <- function(dfGPS,dfY,dfX=NA,kappa,z=1){
#   
#   colsY <- colnames(dfY)
#   #Step 1 - fit linear model and set as prior for betas
#   if(is.data.frame(dfX)){
#     dfXY <- cbind(dfY,dfGPS,dfX)
#     fmla <- as.formula(paste('cbind(',colsY[1],',',colsY[2],')~',paste(colnames(dfX),collapse='+',sep=''),sep=''))
#   } else {
#     dfXY <- cbind(dfY,dfGPS)
#     fmla <- as.formula(paste('cbind(',colsY[1],',',colsY[2],')~1',sep=''))
#   }
#   
#   fit.glm <- glm(fmla,data=dfXY,family=binomial)
#   
#   #Step 2 - fit variogram and set as priors for GP parameters
#   vari <- variog(coords = as.matrix(dfGPS),
#                  data=log((dfY[1]+0.25)/(dfY[2]+0.25)))
#   vari.fit <- variofit(vari, ini.cov.pars = c(2, 0.2),
#                        cov.model = "matern",
#                        fix.nugget = FALSE, nugget = 0 ,
#                        fix.kappa = TRUE, kappa = kappa)
#   
#   sigmaGP <- vari.fit$cov.par[1] 
#   phiLen <- vari.fit$cov.par[2]
#   nugget <- vari.fit$nugget
#   
#   #Step 3 - set priors
#   input.MCMC <- MCMCinput(Y.family = "Binomial", rho.family = "rhoMatern", 
#                           run = 2000*z, run.S = 1, phi.bound = c(phiLen/5,phiLen*5),
#                           priorSigma = "Halft", parSigma = c(sigmaGP, 3),
#                           ifkappa = 0, initials = list(coef(fit.glm),sigmaGP,phiLen,kappa),
#                           scales = c(1.65^2/(dim(dfY)[1])^(1/3),1.65^2/(dim(dfX)[2]+1)^(1/3), 0.5, 0.4, 1))
#   
#   #Step 4 - fit
#   dfXY$nN <- rowSums(dfY)
#   #fmlaGPS <- as.formula(paste('~',paste(colnames(dfGPS),collapse='+')))
#   #fmla <- as.formula(paste('~',paste(colnames(dfX),collapse='+',sep=''),sep=''))
#   #X.mm <- model.matrix(fmla,dfX)
#   browser()
#   mdl.geoCnt <- runMCMC.sf(Y=dfY[,1],L=dfXY$nN,loc=as.matrix(dfGPS),X=as.matrix(dfX),
#                            MCMCinput = input.MCMC,famT=1,n.chn=4,n.cores=4,cluster.type='SOCK')
# }
# 
# #Load Data

# df <- read.csv('/media/pato/DATA/Dev-VAM/HRM/Data/Features/features_all_id_3012_evaluation.csv')
# colnames(df)[which(colnames(df)=='countbyEA')] <- 'n'
# df <- df[df$n>=3,]
# prevCol <- 'prevalence_expenditures' #'prevalence_food'
# df$i <- df$n-round(df$n*df[,prevCol])
# df$j <- round(df$n*df[,prevCol])
# colnames(df)[which(colnames(df)=='i')] <- 'nOK'
# colnames(df)[which(colnames(df)=='j')] <- 'nFI'
# colnames(df)[which(colnames(df)=='gpsLatitude')] <- 'Lat'
# colnames(df)[which(colnames(df)=='gpsLongitude')] <- 'Lng'
# ftrStartIndx <- which(colnames(df)=='X0_x')
# 
# mdl.3012 <- fitPrevGP(df[,c('Lat','Lng')],df[,c('nFI','nOK')],dfX=df[,ftrStartIndx:dim(df)[2]],1.5,method='MCML',z=2)
# df <- read.csv('/media/pato/DATA/Dev-VAM/HRM/Data/Features/features_all_id_3011_scoring.csv')
# ftrStartIndx <- which(colnames(df)=='X0_x')
# prd.3012 <- prdPrevGP(mdl.3012,df[,c('gpsLatitude','gpsLongitude')],df[,c(ftrStartIndx:dim(df)[2])],xCDThresh=c(0.05,0.1,0.2,0.35,0.5),returnDF=TRUE)

# names(df)[1:5] <- c('nOK','nFI','nSurv','Lat','Lng','prevalence')
# #Create splits
# trainIndex = createFolds(c(1:dim(df)[1]),k=5,returnTrain=TRUE)
# 
# #train/predict
# i <- 1
# dfPerf <- data.frame(RMSE=numeric(),
#                      Rsquared=numeric,
#                      MAE=numeric(),
#                      stringsAsFactors=FALSE)
# for(kappa in seq(0.5,2.5,0.25)){
#   predictions <- trainPredictGP(kappa,
#                                 df[trainIndex[[i]],ftrStartIndx:dim(df)[2]],
#                                 df[trainIndex[[i]],c('nOK','nFI')],
#                                 df[trainIndex[[i]],c('Lat','Lng')],
#                                 df[-trainIndex[[i]],ftrStartIndx:dim(df)[2]],
#                                 df[-trainIndex[[i]],c('Lat','Lng')],
#                                 method='auto',z=1,xCDThresh=1-c(0.05,0.1,0.2,0.35,0.5))
#   perf <- evalPerf(predictions,df[-trainIndex[[i]],'prevalence'])
#   dfPerf <- rbind(dfPerf,t(as.matrix(perf)))
# }
# rownames(dfPerf) <- seq(0.5,2.5,0.25)
# 
# #try using geocount
# mdl.geoCnt.fit <- fitGeoCntGP(df[trainIndex[[i]],c('Lat','Lng')],
#                               df[trainIndex[[i]],c('nOK','nFI')],
#                               df[trainIndex[[i]],ftrStartIndx:dim(df)[2]],
#                               1.5,z=1)



