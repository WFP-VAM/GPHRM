# Gaussian Process based High Resolution Prevalence Mapping of Food Security

This is an addendum to [WFP's High Resolution Mapping of Food Security Project](https://github.com/WFP-VAM/HRM). Much of the pipeline is the same except this replaces the estimation functions of HRM with a gaussian process (GP) based approach specifically for estimation of prevalence. In particular, we implement a generalized mixed-model with a logistic-link function, binomial error distribution, and Gaussian spatial process as the stochastic component in a linear predictor. There are several advantages this approach provides over the estimation approaches employed in the current HRM code-base (K-NN and ridge-regression):

  - GPs incorporate spatial correlation from the GPS coordinates of locations and the input features from those locations in a single modelling framework. I.E. all parameters, including coefficients on features and spatial correlation components, are estimated together.
  - GPs naturally account for data with multiple levels of <i>fidelity</i>. This is necessary as our training data are the cluster-level observations of large-scale areal surveys. For example it may be the case that we observe 10 households at a particular location and 8 of 10 fall below the food-security threshold.  However this does not imply that 80% of households at the cluster are food insecure (as would be assumed by K-NN or ridge-regression)!  If we recall the binomial distribution we find that for k=8 and N=10, the underlying prevalence of food-insecurity has a 95% confidence interval between 44% and 97%!  Even doubling the number of observations, as would happen in a high-quality survey, we find that for k=16 and N=20 our equivalent confidence interval is between 56% and 94%.  In real-life scenarios a cluster-level standard deviation that is 10% of the cluster mean is considered excellent! It can often be as high as 20-30% of the cluster mean. GPs allow us to use the training data as it is.
  - As K-NN and ridge-regrssion do not take into the uncertainty of the training data, we cannot reliably understand the uncertainty of the predictions themselves. At best we can know the overall error when performing k-fold cross-val but for any new prediction we would not know the uncertainty the model has about said prediction. With the GP approach we are able to provide a mathematically consistent, analyzable variance that is unique to each prediction. Hence, if predicting into an area where training data is either missing or noisy, or correlation with features is poor, predictions have high variance and vise versa when the opposite holds true. With this information, we are able to inform decision makers when predictions can be trusted and furthermore classify areas into prevalence ranges with a high degree of confidence.
  - finally the performance of the GP approch as measured by prediction error is almost always categorically better than the other aforementioned approaches. Again this is simply because GPs maximixe the knowledge extracted from the data

The disadvantages are of course that modelling is more complicated, takes much longer to train, and the user must have more prior knowledge to set appropriate hyper-parameter choices. The code-base was created for the purpose of allowing a novice to still train the aforementioned model with little to none of the requisite knowledge to set hyper-parameters correctly, using a combination of heuristics and grid-search to self-select these parameters. The core model estimator comes from an existing package [PrevMap: An R Package for Prevalence Mapping by E. Giorgi](https://cran.r-project.org/web/packages/PrevMap/vignettes/PrevMap.pdf).

The code-base is able to:
  - train, predict, and perform k-fold cross-val from training data
  - predict on new data and provide for each prediction location the estimate of prevalence, standard-deviation of estimate, and exceedance-probabilities (probability prevalence exceeds a set of pre-defined thresholds) 
  - create raster image of estimates and raster of standard error for estimates
  
### How to run code (see example):
#### Set-up
Follow setup instructions for HRM.
* 

#### 5-fold cross-val 
```
python GPHRM.py CV args
```
where args are either:
* simply the id of the table "config" in the database
* id and a boolean (T/F) indicating if the input data should be whitened and features reduced through single-lasso variable selection (default is false)
* id and boolean and list of kappa parameters where each kappa parameter (the Matern Kernel shape parameter) is separated with a space -- this parameters you want to select through k-fold cross-val (default is kappaList={0.5,1.5,2.5})

#### train complete model and score
```
python GPHRM.py SCORE args 
```
where args are either:
* `trainID` `scoreID` `kappaParameter` (of the table "config" in the database)
* `trainID` `scoreID` `kappaParameter` `method` where <i>method</i> is the choice between Markov-Chain Mont-Carlo (MCML) and Markov-Chain Maximum-Likelihood (MCML) or 'auto' (see comments in code) default is 'auto'
* `trainID` `scoreID` `kappaParameter` `method` `z` where <i>z</i> controls the number of iterations of the optimizer (see comments in code) default is 'auto'
* `trainID` `scoreID` `kappaParameter` `method` `z` `fitTwice` where <i>fitTwice</i> is a boolean that fits the model twice using the first fit as an informative prior to a second fit, can be set to True, False or 'auto' (default is auto)
* `trainID` `scoreID` `kappaParameter` `method` `z` `fitTwice` `whiten` where <i>whiten</i> is a boolean that indicates if input data should be whitened and features reduced through single-lasso (default is False)

### Work in progress 
Next steps (work in progress) include:
+ automatic selection of the kappa-parameter
+ single codebase in Python only  
+ allowing user to specify (or computer to automatically estimate) a beta prior to the binomial distribution as an input hyper-parameter
  
### Contacts
For more info to collaborate, use or just to know more, you may reach us at gaurav.singhal@wfp.org, jeanbaptiste.pasquier@wfp.org, and lorenzo.riches@wfp.org or submit an issue.
