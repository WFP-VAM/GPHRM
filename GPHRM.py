import pdb

#Create performance evaluation functions
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import KFold

# r square for bayesian prediction - computes distribution of r2 values
def r2_Bayes(prAct,prHat,prStd,seed=42):
    """
    prAct = vector of actual prevalence from dataset
    prHat = vector of model estimated prevalence means
    prStd = vector of model estimates prevalence standard deviation 
    """
    
    prng1 = np.random.RandomState(seed)
    seeds = prng1.randint(low=1,high=10e6,size=1000)
    r2s = np.zeros(1000)
    
    for i in range(1000):
        prng2 = np.random.RandomState(seeds[i])
        prHatS = prng2.multivariate_normal(prHat,np.diag(prStd**2))
        r2s[i] = np.var(prHatS)/(np.var(prHatS)+np.var(prAct-prHatS))
    
    return np.median(r2s)

#standard r2 measures below    
def r2_pearson(y, yhat):
    return stats.pearsonr(y, yhat)[0] ** 2

def R2(y, yhat):
    return r2_score(y, yhat)

#Mean absolute percentage error
def MAPE(y, yhat):
    diff = np.abs(np.divide((y-yhat), y, out=np.zeros_like(yhat), where=y!=0))
    return(np.sum(diff)/len(y))
            
#import from rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP 
from rpy2.robjects.packages import importr
#Activate pandas2ri
pandas2ri.activate()
#increase memory limit for embedded r objects
robjects.r('memory.limit(4096)')

# import R functions - 4 functions imported-------------------------------------------------------------------------------------- # 
# whitenSlct - performs ZCA on training data and selects post-ZCA selects most relevant features (eigenvalues) using single LASSO #
# fitPrevGP  - fits gaussian process model to training data                                                                       #
# prdPrevGP  - given gaussian process model, takes new data and retrieves predictions from model                                  #
# trainPredictGP - convenience function which combines fitPrevGP and prdPrevGP for the purpose of k-fold cross-val                #
# see 'PrevMap an R Package for Prevalence Mapping' by E. Giorgi (https://www.jstatsoft.org/article/view/v078i08)                 #
# ------------------------------------------------------------------------------------------------------------------------------- #
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open('rGPHRM.R', 'r') as f:
    string = f.read()
GPHRM = STAP(string, "GPHRM") # STAP = SignatureTranslatedAnonymousPackage
# imported r functions are now part of STAP object GPHRM

#Whiten and perform LASSO selection on variables
def whitenLASSO(Y,X):
    rX = pandas2ri.py2ri(X)
    rY = pandas2ri.py2ri(Y)
    rXzca = GPHRM.whitenSlct(rY,rX)
    return pandas2ri.ri2py_dataframe(rXzca)

#Create Class for creating gaussian process models
import numpy as np
import pandas as pd
import pickle

class GPmodeller:

    def __init__(self,GPS,Y,X=None,Whiten=False):
        '''
        class to handles the models that use satellite and other features to predict indicator.
        GPS = GPS latitude/longitude for observations in that order
        Y   = should be two columns of Failures/Successes in that order
        X   = additional satellite or other features
        Whiten = perform ZCA and feature selection on features (True/False)
        '''
        self.GPS = GPS
        self.Y = Y
        
        if Whiten:
            self.X = whitenLASSO(Y,X)
        else:
            self.X = X

    #save pickled object
    def objSave(self,f):
        with open(f,'wb') as handle:
            pickle.dump(self, handle)
        
        return True
    
    #load instance from pickled object
    @classmethod
    def objLoad(cls,f):
        return pickle.load(open(f,"rb"))
    
    #performs k-fold cross val        
    @staticmethod
    def _k_fold_cross_validation(X,K,n,seed=42):
        '''
        X = training data pandas dataframe
        K = number of cross-vals
        n = number of random-restarts of k cross-vals, i.e. cross-val performed k*n times
        '''

        j = 0
        prng = np.random.RandomState(seed)
        
        for i in prng.randint(low=1,high=10e6,size=n):
            X = X.sample(frac=1, random_state=i)
            
            for k in range(K):
                j += 1
                training = [x for i, x in enumerate(X.index) if i % K != k]
                validation = [x for i, x in enumerate(X.index) if i % K == k]
                yield training, validation, j

    @staticmethod
    def trainPredict(kappa,dfTrnY,dfTrnX,dfTstX,dfTrnGPS,dfTstGPS,method='auto',z='auto',xCDThresh=[0.05,0.1,0.2,0.35,0.5]):
        """
        combines train and test in a single call for convenience when doing k-fold CV -> returns predictions on test data
        kappa    = shape parameter for GP Matern kernel
        dfTrnX   = training data features
        dfTrnY   = training data targets (two columns: |n successes|n failures|)
        dfTrnGPS = training data GPS coordinates (two columns: |lat|lng|)
        dfTstX   = test data features
        dfTstGPS = test data GPS coordinates (two columns: |lat|lng|)
        method   = method to train model, can choose either
                   'Bayes' which uses the Markov Chain Monte Carlo approach (slow but robust)
                   'MCML'  which uses Markov Chain Maximum Likelihood approach (faster but more prone to error)
                   'auto'  which heuristically chooses between the two
        z        = sets Markov Chain optimizer parameters which by is set to
                   iterations = max(2000,5000*z) and burn = max(1000,1000*z)
        xCDThresh= exceedance thresholds, predictions include probabilities prevalence exceeds the
                   defined thresholds in this vector

        """

        #Convert from pandas dataframe to R dataframe trainX,trainY,trainGPS,testX,testGPS
        rdfTrnY = pandas2ri.py2ri(dfTrnY)
        rdfTrnX = pandas2ri.py2ri(dfTrnX)
        rdfTstX = pandas2ri.py2ri(dfTstX)
        rdfTrnGPS = pandas2ri.py2ri(dfTrnGPS)
        rdfTstGPS = pandas2ri.py2ri(dfTstGPS)
        
        #convert exceedance thresholds
        rXCDThresh = robjects.vectors.FloatVector(xCDThresh)
        
        # ------------------------------------------------------------------------------------ #
        # Call R function and return results                                                   #
        # R function Arguments: kappa,trainX,trainY,trainGPS,testX,testGPS,method,z, xCDThresh #
        # ------------------------------------------------------------------------------------ #
        rslt = GPHRM.trainPredictGP(kappa,rdfTrnX,rdfTrnY,rdfTrnGPS,rdfTstX,rdfTstGPS,method,z,rXCDThresh)
        dfRslt = pandas2ri.ri2py_dataframe(rslt)
        
        return dfRslt

    def train(self,kappa,method='Bayes',z=2,fitTwice='auto'):
        """
        trains model from  intiial X, Y, GPS dataframes that the object was instantiated with
            -> returns self with model as attribute
        kappa    = shape parameter for GP Matern kernel
        method   = method to train model, can choose either
                   'Bayes' which uses the Markov Chain Monte Carlo approach (slow but robust)
                   'MCML'  which uses Markov Chain Maximum Likelihood approach (faster but more prone to error)
                   'auto'  which heuristically chooses between the two
        z        = sets Markov Chain optimizer parameters which by is set to
                   iterations = max(2000,5000*z) and burn = max(1000,1000*z)
        fitTwice = for final training we recommend fitting the model twice, using the estimated parameters
                   from the initial fit as priors for the second fit, can be set to True, False, or 'auto'

        """

        #Convert from pandas dataframe to R dataframe trainX,trainY,trainGPS
        rdfTrnY = pandas2ri.py2ri(self.Y)
        rdfTrnX = pandas2ri.py2ri(self.X)
        rdfTrnGPS = pandas2ri.py2ri(self.GPS)

        # ------------------------------------------------------------------- #
        # Call R function and return results                                  #
        # R function Arguments: kappa,trainX,trainGPS,testX,method,z,fitTwice #
        # ------------------------------------------------------------------- #
        self.model = GPHRM.fitPrevGP(rdfTrnGPS,rdfTrnY,rdfTrnX,kappa,method=method,z=z,fitTwice=fitTwice)
        self.z = z
        return self

    def predict(self,dfPrdX,dfPrdGPS,xCDThresh=[0.05,0.1,0.2,0.35,0.5]):
        """
        performs predictions on new data given model -> returns self with new predictions as an attribute
        dfTstX   = test data features
        dfTstGPS = test data GPS coordinates (two columns: |lat|lng|)
        xCDThresh= exceedance thresholds, predictions include probabilities prevalence exceeds the
                   defined thresholds in this vector
        """
        
        self.prdX = dfPrdX
        self.prdGPS = dfPrdGPS
        
        #Convert from pandas dataframe to R dataframe prdX,prdGPS
        rdfPrdX = pandas2ri.py2ri(dfPrdX)
        rdfPrdGPS = pandas2ri.py2ri(dfPrdGPS)
        
        #convert exceedance thresholds
        rXCDThresh = robjects.vectors.FloatVector(xCDThresh)

        # --------------------------------------------------- #
        # Call R function and return results                  #
        # R function Arguments: model,testX,testGPS,xCDThresh #
        # --------------------------------------------------- #
        rslt = GPHRM.prdPrevGP(self.model,rdfPrdGPS,rdfPrdX,rXCDThresh,returnDF=True)
        self.predictions = pandas2ri.ri2py_dataframe(rslt)
        
        return self 


    def evaluate(self,k,n,kappa_list,seed=42):
        """
        Performs k-fold cross-val across all values of kappa in kappa_list
        k = number of cross-vals
        n =  number of random-restars for cross-vals
        kappa_list = vector of kappa parameters to try
        """
        
        self.kappa_list = kappa_list
        self.scores = pd.DataFrame(np.full([len(kappa_list)*k*n,9],np.nan),
                                   index=[np.repeat(list(range(1,k*n+1)),len(kappa_list)),np.tile(kappa_list,k*n)],
                                   columns=['r2','r2_Scaled','r2_Bayes','r2_Tjur','MAPE', \
                                            'GOF_Pearson','GOF_logLklh','prHat','prAct'])
        self.scores['prHat'] = self.scores['prHat'].astype(object)
        self.scores['prAct'] = self.scores['prAct'].astype(object)
        
        inner_cv = KFold(5,shuffle=True,random_state=seed)

        print('-> grid searching and cross validation ...')

        for trnIndx, tstIndx, j in self._k_fold_cross_validation(self.GPS,k,n,seed):

            print('Commenced Loop '+str(j))
            
            gpsTrn = self.GPS.loc[trnIndx,:]
            gpsTst = self.GPS.loc[tstIndx,:]
            xTrn = self.X.loc[trnIndx, :]
            xTst = self.X.loc[tstIndx,:]
            yTrn = self.Y.loc[trnIndx,:]
            
            yTst = self.Y.loc[tstIndx,:]
            #Convert to prevalences
            prTst = (yTst.iloc[:,0]/yTst.sum(1)).values
            #Number of observations per cluster
            nTst = yTst.sum(1).values
            #degrees of freedom
            ddof = yTst.shape[0]-1
            #Successes per cluster
            yAct = yTst.iloc[:,0].values

            for kappa in self.kappa_list:
                print('Training model for kappa='+str(kappa))
                
                try:
                    rslt = self.trainPredict(kappa,yTrn,xTrn,xTst,gpsTrn,gpsTst)
                    #probabilities
                    prHat = rslt.iloc[:,0].values
                    #multiplied by observations per cluster
                    yHat = prHat*yTst.sum(1).values
                    
                    #Normal R2 - not applicable for probabilities/binomial outcomes                    
                    self.scores.loc[(j,kappa),'r2'] = np.abs(R2(prTst,prHat))
                    #Scaled by observations per cluster (conditional)
                    self.scores.loc[(j,kappa),'r2_Scaled'] = np.abs(R2(yAct,yHat))
                    #Unconditional R2 for Bayesian Models (http://www.stat.columbia.edu/~gelman/research/unpublished/bayes_R2.pdf)
                    self.scores.loc[(j,kappa),'r2_Bayes'] = r2_Bayes(prTst,prHat,rslt['stdError'].values)
                    #Tjur's R2 for binomial outcomes (https://support.sas.com/kb/39/109.html)
                    self.scores.loc[(j,kappa),'r2_Tjur'] =  np.abs(np.sum(yAct*prHat)/np.sum(nTst)- \
                                                                   np.sum(yTst.iloc[:,1].values*prHat)/np.sum(nTst))
                                                            #np.sum((yAct-yTst.iloc[:,1].values)*prHat)/np.sum(nTst)
                    self.scores.loc[(j,kappa),'MAPE'] = MAPE(yTst.iloc[:,0].values,yHat)
                    #Goodness of fit measures for probabilistic categorical data 
                    self.scores.loc[(j,kappa),'GOF_Pearson'] = stats.chi2.cdf(np.sum((yAct-yHat)**2/yHat),ddof)
                    self.scores.loc[(j,kappa),'GOF_logLklh'] = stats.chi2.cdf(2*np.sum(yAct*np.log((yAct+1e-6)/(yHat+1e-6))),ddof)
                    #Save prHat, prAct
                    self.scores.at[(j,kappa),'prHat'] = prHat
                    self.scores.at[(j,kappa),'prAct'] = prTst
                except Exception as e:
                    print(e)
                
                print('Finished Training -- results:')
                print(self.scores.loc[(j,kappa),:])
            
            print(self.scores)
            
        return self 

import os
import sys
import yaml
import sqlalchemy
from sqlalchemy import Column, Integer, Float, String, create_engine  
from sqlalchemy.dialects.postgresql import JSON, JSONB

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import HRM

from HRMutils import tifgenerator
from HRMutils import aggregate
from HRMutils import points_to_polygon

def score(trainID,scoreID,kappa,method='Bayes',z=2,fitTwice='auto',whiten=False):
    """
    function to train and score model given dataset IDs from database and training parameters
    trainID = overloaded parameter-either 
             (a) database ID of training dataset 
             (b) filepath for already trained model (as a pickled GPModeller object)
    scoreID = databaise ID for prediction dataset
    kappa    = shape parameter for GP Matern kernel
    method   = method to train model, can choose either
               'Bayes' which uses the Markov Chain Monte Carlo approach (slow but robust)
               'MCML'  which uses Markov Chain Maximum Likelihood approach (faster but more prone to error)
               'auto'  which heuristically chooses between the two
    z        = sets Markov Chain optimizer parameters which by is set to
               iterations = max(2000,5000*z) and burn = max(1000,1000*z)
    fitTwice = for final training we recommend fitting the model twice, using the estimated parameters
               from the initial fit as priors for the second fit, can be set to True, False, or 'auto'
    Whiten = perform ZCA and feature selection on features (True/False)

    """

    if ~isinstance(whiten,bool):
        whiten = str(whiten) in ['TRUE','True','true','t','T', '1','Yes','Y','y','yes','YES','yeah','yup', 'certainly', 'uh-huh']

    # ----------------- #
    # SETUP #############
    # ----------------- #
    if isinstance(trainID,str) and os.path.isfile(trainID):
        #load pre-existing model object from file if already trained
        mdlGPHRM = GPmodeller.objLoad(trainID)
    else:
        print(str(np.datetime64('now')), " INFO: config id =", id)

        #Connect to database
        with open('../HRM/private_config.yml', 'r') as cfgfile:
            private_config = yaml.load(cfgfile)

        engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                               .format(private_config['DB']['user'], private_config['DB']['password'],
                                       private_config['DB']['host'], private_config['DB']['database']))
        
        config = pd.read_sql_query("select * from config_new where id = {}".format(trainID), engine)
        dataset = config.get("dataset_filename")[0]
        indicator = config["indicator"][0]
        rasterFile = config["satellite_grid"][0]
        aggregate_factor = config["base_raster_aggregation"][0]
        scope = config["scope"][0]

        # ----------------------- #
        # Read Train Data & Train #
        # ----------------------- #
        filepath = '../HRM/Data/Features/features_all_id_'+str(trainID)+'_evaluation.csv'
        print('read training file '+filepath)

        df = pd.read_csv(filepath)
        #filter to make sure number of samples>3
        df = df.rename(columns={'countbyEA':'n'})
        df = df.loc[df['n']>=3]

        #create variables for positive/negative (success/failure) binomial trials
        df['i'] = np.round(df['n']*df[indicator])
        df['j'] = df['n']-df['i']
        df = df.rename(columns={'i':'nPos','j':'nNeg','n':'nSurv','gpsLatitude':'Lat','gpsLongitude':'Lng'})
        featStrtIndx = df.columns.get_loc("0_x")

        #train model and save
        print('training GPHRM model')
        mdlGPHRM = GPmodeller(df.loc[:,['Lat','Lng']],df.loc[:,['nPos','nNeg']],df.iloc[:,featStrtIndx:],Whiten=whiten).train(kappa,method=method,z=z,fitTwice=fitTwice)
        mdlFilepath ='savedModels/GPHRMmdl_id_'+str(trainID)+'_kappa_'+str(kappa)+'_'+('MCMC' if method=='Bayes' else 'MCML')+'_iter_'+str(5000*z)+'.pkl'
        mdlFilepath = os.path.join(os.getcwd(),mdlFilepath)
        mdlGPHRM.objSave(mdlFilepath)
        print('Finished training. Output saved at '+filepath)

    # --------------------------- #
    # Read Scoring Data & Predict #
    # --------------------------- #
    if scoreID is not None:

        filepath = '../HRM/Data/Features/features_all_id_'+str(scoreID)+'_scoring.csv'
        df = pd.read_csv(filepath)
        df = df.rename(columns={'gpsLatitude':'Lat','gpsLongitude':'Lng'})
        print('read file '+filepath+' now scoring on model')

        #score model
        featStrtIndx = df.columns.get_loc("0_x")
        mdlGPHRM = mdlGPHRM.predict(df.iloc[:,featStrtIndx:],df.loc[:,['Lat','Lng']])
        rsltDF = pd.concat([df.loc[:,['i','j','Lat','Lng']].reset_index(drop=True), mdlGPHRM.predictions.reset_index(drop=True)], axis=1)
        rsltDF = rsltDF.rename(columns={'Lat':'gpsLatitude','Lng':'gpsLongitude'})
        print('finished scoring - writing results to database')

        # --------------------------- #
        # Write results to database   #
        # --------------------------- #
        engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                               .format(private_config['DB']['user'], private_config['DB']['password'],
                                       private_config['DB']['host'], private_config['DB']['database']))

        # Prepare SQL
        rsltTbl = sqlalchemy.table("gp_predict",Column('train_config_id', Integer),
                                                Column('score_config_id', Integer),
                                                Column('model_save_file', String),
                                                Column('hyperparams',JSON),
                                                Column('predict_gps', JSON),
                                                Column('predict_score', JSON),
                                                Column('predict_error',JSON),
                                                Column('exceednc_prob',JSON))

        #compile insert statement
        statement = rsltTbl.insert().values(train_config_id = trainID,
                                            score_config_id = scoreID,
                                            model_save_file = os.path.basename(mdlFilepath),
                                            hyperparams = {'kappa':kappa,'method':method,'z':z},
                                            predict_gps = rsltDF.iloc[:,[0,1,2,3]].to_json(),
                                            predict_score = rsltDF.iloc[:,[0,1,4]].to_json(),
                                            predict_error = rsltDF.iloc[:,[0,1,5,6,7,8,9,10]].to_json(),
                                            exceednc_prob = rsltDF.iloc[:,[0,1,11,12,13,14,15]].to_json()
                                           )
        #Execute insert
        engine.execute(statement)
        print('finished writing results to database - creating raster')

        # --------------------------- #
        # create tif                  #
        # --------------------------- #
        outfile = "/Results/scalerout_{}_prevalence.tif".format(scoreID)
        tifgenerator(outfile, lowResRstrFile, rsltDF.iloc[:,[0,1,4]], value='prevalence')
        outfile = "/Results/scalerout_{}_stdErr.tif".format(scoreID)
        tifgenerator(outfile, lowResRstrFile, rsltDF.iloc[:,[0,1,5]], value='stdError')
        print('finished generating tif - located at:'+outfile)

def run(id,whiten=False,kappaList=[0.5,1.5,2.5]):
    """
    function to perform 5-fold cross-val on each value of kappaList given a dataset from the database
    id = Database ID for training data
    whiten = perform ZCA and feature selection on features (True/False)
    kappaList = list of kappa values to try
    """

    if ~isinstance(whiten,bool):
        whiten = str(whiten) in ['TRUE','True','true','t','T', '1','Yes','Y','y','yes','YES','yeah','yup', 'certainly', 'uh-huh']

    # ----------------- #
    # SETUP #############
    # ----------------- #

    print(str(np.datetime64('now')), " INFO: config id =", id)

    #Connect to database
    with open('../HRM/private_config.yml', 'r') as cfgfile:
        private_config = yaml.load(cfgfile)

    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))
    
    config = pd.read_sql_query("select * from config_new where id = {}".format(id), engine)
    dataset = config.get("dataset_filename")[0]
    indicator = config["indicator"][0]
    raster = config["satellite_grid"][0]
    aggregate_factor = config["base_raster_aggregation"][0]
    scope = config["scope"][0]

    # ----------------- #
    # Read Data & Eval  #
    # ----------------- #
    filepath = '../HRM/Data/Features/features_all_id_'+str(id)+'_evaluation.csv'
    print('read file '+filepath)

    df = pd.read_csv(filepath)
    #filter to make sure number of samples>3
    df = df.rename(columns={'countbyEA':'n'})
    df = df.loc[df['n']>=3]

    #create variables for positive/negative (success/failure) binomial trials
    df['i'] = np.round(df['n']*df[indicator])
    df['j'] = df['n']-df['i']
    df = df.rename(columns={'i':'nPos','j':'nNeg','n':'nSurv','gpsLatitude':'Lat','gpsLongitude':'Lng'})
    featStrtIndx = df.columns.get_loc("0_x")

    #Train/Evaluate
    print('Detected binomial variable '+indicator+' -> commencing GPHRM training')
    nn = 1 if len(kappaList)>1 else 3
    gpHRMeval = GPmodeller(df.loc[:,['Lat','Lng']],df.loc[:,['nPos','nNeg']],df.iloc[:,featStrtIndx:],Whiten=whiten).evaluate(5,nn,kappaList)
    rsltDF = gpHRMeval.scores.dropna(axis=0)

    # ------------------ #
    # write scores to DB #
    # ------------------ #
    print('Finished Evaluation - writing results to DB')
    engine = create_engine("""postgresql+psycopg2://{}:{}@{}/{}"""
                           .format(private_config['DB']['user'], private_config['DB']['password'],
                                   private_config['DB']['host'], private_config['DB']['database']))

    # Prepare SQL
    rsltTbl = sqlalchemy.table("gp_results",Column('config_id', Integer),
                                            Column('hyperParams', JSON),
                                            Column('performance', JSON),
                                            Column('r2_x',Float ),
                                            Column('MAPE',Float),
                                            Column('prdVsAct',JSON))

    #enumerate by hyperparameters
    for kappa in kappaList:

        #subset dataframe
        kpDF = rsltDF.xs(kappa,level=1)

        #compile all predicted vs actual into single dictionary
        prdVsActDict = {}
        for index,row in kpDF.iterrows():
            d = dict(zip(row['prHat'],row['prAct']))
            for k, v in d.items():  # d.items() in Python 3+
                k = np.round(k,4)
                v = np.round(v,4)
                prdVsActDict.setdefault(k, []).append(v)

        #compile insert statement
        statement = rsltTbl.insert().values(config_id = id,
                                            hyperParams = {'kappa':kappa},
                                            performance = {'r2':kpDF['r2'].mean().round(4),
                                                           'r2_Scaled': kpDF['r2_Scaled'].mean().round(4),
                                                           'r2_Bayes': kpDF['r2_Bayes'].mean().round(4),
                                                           'r2_Tjur': kpDF['r2_Tjur'].mean().round(4),
                                                           'MAPE': kpDF['MAPE'].mean().round(4),
                                                           'GOF_Pearson': kpDF['GOF_Pearson'].mean().round(4),
                                                           'GOF_logLklh': kpDF['GOF_logLklh'].mean().round(4),
                                                          },
                                            r2_x = kpDF['r2_Bayes'].mean().round(4),
                                            MAPE = kpDF['MAPE'].mean().round(4),
                                            prdVsAct = prdVsActDict
                                           )
        #Execute insert
        engine.execute(statement)

    print('Finished entering results -> terminating')

#performs run function if executed as a direct call
if __name__ == "__main__":

    if sys.argv[1]=='CV':
        if len(sys.argv)==3:
            run(sys.argv[2])
        elif len(sys.argv)==4:
            run(sys.argv[2],sys.argv[3])
        elif len(sys.argv)>5:
            run(sys.argv[2],sys.argv[3],sys.argv[4:])
    else:
        if len(sys.argv)==5:
            score(sys.argv[2],sys.argv[3],sys.argv[4])
        elif len(sys.argv)==6:
            score(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        elif len(sys.argv)==7:
            score(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
        elif len(sys.argv)==8:
            score(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])
        elif len(sys.argv)==9:
            score(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])
