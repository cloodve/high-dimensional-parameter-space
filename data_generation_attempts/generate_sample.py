import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import random
import scipy.stats as stats
import scipy
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import seaborn as sns
sns.set()


np.random.seed(52)
# new_seeds = np.random.randint(1000000)

def generateNormalVariables(n, p):

    X = {} # initialize dictionary to hold values

    for j in range(p):
        #np.random.seed(seed = j)   #seed for distribution
        #random.seed(j)   #seed for random number generator
        X[j] = np.random.normal(0, 1/np.sqrt(n), n) # NOTE: ADDED 1/SQRT(N) SCALING 3/31
    return X

def generateX(n,p):
    
    # Initialize dataframe with an intercept column
    #intercept = np.repeat(1,n) # Create n 1's for intercept column
    #X = pd.DataFrame(intercept.reshape(n,1))
    # COMMENTED OUT INTERCEPT FOR TIME BEING. CAUSING ISSUES WITH SIGNAL STRENGTH SCALING
    X = pd.DataFrame()
    
    # Generate data
    if p != 0:
        Xnorm = generateNormalVariables(n, p)
        Xnorm = pd.DataFrame.from_dict(Xnorm)
        X = pd.concat([X, Xnorm],ignore_index=True,axis=1)
    
    return X

def generateRandomBeta(q, mu, stdev):
    beta = {}
    for j in range(q):
        #random.seed(j)#seed for random number generator
        beta[j] = np.random.normal(mu,stdev)
        #beta[j] = np.random.poisson(stdev)-3 # Checking to see what happens when betas come from noncentral poisson
        #beta[j] = np.random.uniform(mu,stdev)
    
    beta = pd.DataFrame(list(beta.items()))
    beta = beta.drop([0],axis=1)
    
    return beta

def sigmoid(z):
    return 1/(1+np.exp(-z))

def generateResponseVariable(X, beta, dist):
    
    beta = np.squeeze(beta)
    
    if dist == 'bernoulli':
        meanValues = sigmoid(X.dot(beta)) 
    elif dist == 'poisson':
        meanValues = np.exp(X.dot(beta))
    elif dist == 'exponential':
        meanValues = 1/(X.dot(beta))
    else:
        print('please spell check distribution name, all lowercase: bernoulli, poisson or exponential')
        
    y = []
    
    #np.random.seed(123)
    for eachMean in np.squeeze(meanValues.values):
        if dist == 'bernoulli':
            randomPrediction = np.random.binomial(1,eachMean)
        elif dist == 'poisson':
            randomPrediction = np.random.poisson(eachMean)
        elif dist == 'exponential':
            randomPrediction = np.random.exponential(eachMean)
        y.append(randomPrediction)
        #print(randomPrediction, eachMean)
    
    return meanValues, y

def generateData(dist, n, p, mu, stdev, signal_strength):
    
    # Generate Data
    X = generateX(n=n, p=p)

    # Generate Betas
    q = X.shape[1]
    beta = generateRandomBeta(q=q, mu=mu, stdev=stdev)
    beta_norm_squared = beta.T.dot(beta).values[0,0]
    beta_scaled = beta * np.sqrt((n * signal_strength) / beta_norm_squared)

    # Generate Response Variable (and associated means - what goes into the link fn)
    means, y = generateResponseVariable(X=X, beta=beta_scaled, dist=dist) ### dist means pass distribution name as string
    
    # Make sure we return numpy arrays (easier to work with later and all names are useless here anyway)
    beta_scaled = np.squeeze(np.array(beta_scaled))
    X = np.array(X)
    y = np.array(y)    
    
    return X, beta_scaled, y, means




def generate_sample(n, mu=10, covariates=None):
    # number of observations
    # n=200

    # number of (Gaussian) covariates
    # p=200# int(n/5)
    if not covariates:
        p = int(n / 5)
    else: 
        p = covariates

    # signal strength
    signal_strength = 5

    # Parameters for Distribution to draw Betas from. Betas ~ N(mu, stdev^2)
    # EF: mu = mu
    stdev = 1.0

    # Generate the data
    # - X is an (n,p+1) dimensional array with the n rows corresponding to observations and the p+1 columns
    #   corresponding to the covariates + a column of 1's (for the intercept term)
    # - Beta is the (p+1,) dimensional array of 'True' regression coefficients
    # - y is the (n,) dimensional array of response values for the observations
    # - means is the (n,) dimensional array of predicted values (probabilities in case of logistic)
    # *THOUGHT: Perhaps we could use another term instead of means to avoid confusion with the systematic component?
    # * perhaps y_hat, preds, etc?
    X, Beta, y, means = generateData(dist = 'bernoulli', n = n, p = p, mu = mu, stdev = stdev, 
                                    signal_strength = signal_strength) 


    df = pd.DataFrame(X)
    y = np.array(y)
    # y = np.where(y>0,1,-1)# Changed label 0 to -1, For convenience, we choose the label as {-1,1} instead of {0,1}
    df['y'] = pd.Series(y)
    return df



def plot_betas(betas):
    plt.scatter(list(range(len(betas))), betas)
    plt.show()


# df = generate_sample(1000, 10)

#################################################################################### - START
## Summary - Start - Unable to reproduce the desired conditions as noted in the paper.
## Unable to get the probabilities to fall in a particular range


# df = generate_sample(1000, 10)
# # df2 = generate_sample(250, mu=0)
# # df3 = generate_sample(250, mu=-10)

# # df.append(df2)
# # df.append(df3)


# print(df)
# print(df.shape)

# X_train, X_test, y_train, y_test = train_test_split(df.loc[:,~df.columns.isin(['y'])], df.y.to_list(), test_size=0.25)
# log_reg = sm.Logit(y_train,X_train).fit()
# print(log_reg.summary())

# # plot_betas(log_reg.params.values)
# y_pred = list(map(round, log_reg.predict(X_test)))
# y_probs = log_reg.predict(X_test)
# ret = np.where(np.logical_and(y_probs >= 0.11, y_probs <= 0.989))

# # print(ret[:5])
# # print(len(ret[0])/len(X_test))

# ret2 = np.matmul(X_test, log_reg.params)
# ret3 = np.where(np.logical_and(ret2 >= -4.472, ret2 <= 4.472))

# # print(ret2[:10])
# # print(len(ret3[0])/len(X_test))
# # y_pred = list(map(round, log_reg.predict_prob(X_test)))
# # print(y_pred[:5])
# # print(y_probs[:5])

# # print(confusion_matrix(y_test, y_pred))
# # print(accuracy_score(y_test, y_pred))
# # print(np.mean(log_reg.params))
# # plot_betas(log_reg.params)
# # print(np.trace(confusion_matrix(y_test, y_pred))/len(y_pred))
#################################################################################### - END



#################################################################################### - START
# Failure - Cannot generate two datasets for big model and small model... Otherwise
#           you get several NaNs. I.E. Data has to be from same source.

# https://www.statology.org/likelihood-ratio-test-in-python/
# https://learning.oreilly.com/library/view/data-science-revealed/9781484268704/html/506148_1_En_5_Chapter.xhtml
# Create multiple random numbers
# Generate the sample of 1000 multiple times
# Fit a model and sample 100 points of LLR
# Plot the the points and see if it looks like ChiSquare
np.random.seed(52)
sample_size = 1000
new_seeds = np.random.randint(10000000, size=5000)
data_points = []
p_vals = []

start_time = time.time()
for count, seed in enumerate(new_seeds):
    loop_start_time = time.time()

    np.random.seed(seed)
    ## ********* Below looks Chi-Squared *********
    # samp2 = generate_sample(1000, covariates=250)
    samp2 = generate_sample(200, covariates=2)
    ##
    # Try to dynamically create bins
    # Try to dynamically create bins for kl-divergence
    # get more samples per bin

    
    # samp2 = generate_sample(sample_size, covariates=25)
    
    # samp2 = sm.add_constant(samp2)
    # print(samp2.head())

    # X2_train, X2_test, y2_train, y2_test = train_test_split(samp2.loc[:,~samp2.columns.isin(['y'])], samp2.y.to_list(), test_size=0.25)
    # print(samp2.loc[:,~samp2.columns.isin(['y',3, 4, 5, 6, 7, 8])])

    log_reg1 = sm.Logit(samp2.y.to_list(), samp2.loc[:,~samp2.columns.isin(['y', 1])]).fit(disp=0)
    log_reg2 = sm.Logit(samp2.y.to_list(), samp2.loc[:,~samp2.columns.isin(['y'])]).fit(disp=0)

    lrt = -2 * (log_reg1.llf - log_reg2.llf)
    data_points.append(lrt)
    p_vals.append(scipy.stats.chi2.sf(lrt, 1))

    if count % 10 == 0:
        end_time = time.time()
        print(f'Count: {count} Since Start: {end_time - start_time} Current Loop: {end_time - loop_start_time}')

# print(dir(log_reg1))
# print(log_reg1.summary())
# print(log_reg2.summary())
# # fig, ax = plt.subplots(figsize =(10, 7))
# # ax.hist(p_vals, bins=25)
# # plt.show()
fig, ax = plt.subplots(figsize =(10, 7))
pd.qcut(data_points, 10)
ax.hist(data_points, bins=30)
plt.title('LRT Distribution')
plt.show()


# mapped_points = list(map(int, data_points))
# df = pd.DataFrame({'points':mapped_points})
# values = df['points'].value_counts().index.tolist()
# counts = df['points'].value_counts().tolist()
# print(df['points'].value_counts(), values, counts)
# plt.hist(values, bins=range(len(values)), weights=counts )
# plt.show()

# x = np.linspace(stats.chi2.ppf(0.01, 2),
#                 stats.chi2.ppf(0.99, 2), 100)

# x = np.arange(0, 20, 0.001)

# #define multiple Chi-square distributions
# plt.plot(x, stats.chi2.pdf(x, df=2), label='df: 2')
# plt.show()



#################################################################################### KL - START
# data_points = [0.37536407494488344, 1.573011783802258, 3.7805338543171274, 8.050094666534108, 0.10175310341918475, 4.273815137862357, 0.02068448486895136, 3.4611951106194, 0.000359459525782313, 4.160744926356443, 10.36418627520959, 0.7894130823751766, 0.6529905331778139, 9.448478957785198, 3.9611027043667093, 0.5038474805666908, 1.771235340038757, 0.3551884383952597, 0.1562700658300571, 1.246525436012746, 7.3442174708930565, 5.419039089274719, 3.8128513564461173, 0.9839549747379408, 4.730801283150839, 1.71504293747563, 2.942931682614301, 4.15255933530932, 10.815009455930465, 9.840115447443878, 6.132279828893445, 7.8550576358121305, 14.196539148273303, 34.67738856864864, 0.5375946904544264, 0.5269339136809776, 0.23728002887574462, 0.8521495320540566, 0.10009882288287031, 18.097061073239104, 5.911744122785535, 12.06297205797378, 2.764515246328557, 8.65172675181475, 11.283978535130814, 6.518771518827663, 2.641427465122149, 2.6522551498356677, 8.978518032915389, 6.959460433477915, 8.17015889475607, 3.3355115819173875, 2.4654001229457663, 0.7842089804897796, 16.32687882994415, 3.2210545937643644, 12.212600965767393, 4.314933381432098, 18.840268061803414, 0.08768895347753869, 10.524972147767983, 1.7917100523884528, 2.828700551451817, 9.022228510308622, 1.8727491124650726, 4.125015017353874, 10.591736873178405, 15.497466670927906, 0.9833145670056638, 5.213052626805279, 1.0077110693094369, 0.19422848714546603, 1.6789025933159962, 7.763712635459967, 8.624695907319904, 0.03886834928394478, 7.984146447424706, 10.713725957378841, 2.349643432461619, 10.099452486439276, 0.0592185535531371, 0.3716016501985564, 1.5299790416470387, 4.425609832500868, 9.543330726555695, 2.0662701650730924, 1.9008001509583323, 11.848179258438478, 0.14029233835648824, 0.858589509852834, 0.00014620571403156646, 1.714650676455193, 1.3411512606739677, 8.079144396699036, 10.326487385041105, 5.36013577033961, 15.91930246851689, 5.5031717356652905, 6.921330862138348, 7.027679976255342, 6.698304522553087, 2.9903500135174994, 0.2618540110970571, 3.211456456032181, 4.494907639249078, 0.02667237505127673, 3.3532766882282488, 12.580809025215416, 7.773198490460203, 5.740949035045048, 6.13823545554223, 1.4040985560755246, 7.75946906967917, 29.484706184807095, 0.03683333846458936, 18.71082535252941, 5.681898751825912, 11.057231246416649, 5.941154028784979, 2.564572720463474, 18.397590691076985, 6.958974986799944, 0.16110197289364692, 15.864826063170725, 0.8024491790066008, 1.569663838543221, 8.30799763397701, 7.722031514209931, 3.5011796766343366, 5.909969430514508, 7.021267611079253, 1.437493748098177, 7.4990317091496195, 21.673256121794964, 8.924912126212632, 13.778357214272887, 1.8934359998241348, 1.045668735588123, 11.482525822222328, 0.4343842089575958, 0.22747342044777952, 12.875737246281062, 5.805454683594263, 4.3443039459849615, 0.9341779273617021, 0.0015612036004597485, 8.051446365322448, 20.14796807428155, 1.3035879470234306, 11.381566215292679, 0.702735500686714, 8.145143329932978, 0.031336016401326106, 4.5275856475079195, 14.275992977349006, 8.247151761548764, 1.4460229891443248, 0.6370734736319719, 4.793315038431672, 6.097800016831513, 1.4254493602624336, 7.535134309761588, 3.638247019971118, 0.19575310689020853, 5.410369954538538, 9.803237671956566, 2.7928280911281718, 7.131224304136026, 3.6052891308275434, 1.4367373512310166, 3.744884523869416, 6.20389164878776, 8.650871884431723, 8.126065324033647, 9.37237802363444, 1.4389657554641815, 0.9768533169858529, 6.6872237451866, 19.058445747701825, 2.2319712190052883, 2.6619743843666015, 0.02821311645084279, 10.18443769930974, 21.8387038686473, 4.632187949117764, 2.4288373435987296, 2.8945142772115275, 0.09000511274845735, 3.5946434713681583, 1.5971752105770065e-05, 3.276635518815283, 0.881408205475168, 33.18108224843235, 1.9490678537795247, 0.1336914353630334, 18.84596574320426, 0.7028250332119796, 2.169607594870172, 10.439261303057691, 2.6973303676747946, 4.30058005673834, 8.56374778778192, 2.430460701409004, 15.573241129158589, 1.151811424938785, 1.0033356133849622, 0.42898659149312834, 0.8002772511354976, 0.4488379872564394, 0.09303094822132607, 0.7597458730305675, 10.766638580761082, 5.536185987930935, 2.504337331660736, 0.6959031018147925, 0.865103656738313, 1.2699019035717356, 5.842905465887526, 8.134254072606382, 10.236203107930976, 0.9708088192143975, 0.04576763088471125, 7.008643262504819, 1.9906820330161281, 5.106149841410186, 9.83893163433757, 13.245055034556913, 0.010998360742490831, 0.5987914714892497, 18.326866007181906, 10.319901756301363, 1.6205796180634309, 0.7314764182865758, 7.5838003954056035, 1.6593021688099867, 7.834374648392384, 6.2245644765718, 2.16105421418618, 0.5075413506633595, 5.772433110382323, 27.292109757708715, 0.8527805008551752, 11.20682596386365, 1.5297756103939832, 0.01576491158789395, 0.1183025153639079, 7.5920480389335125, 15.469968224077945, 0.41662970144838596, 3.3902933596128264]
# print(data_points)
bs = 90
total_count = len(data_points)
bins = list(range(1, bs))
digitized = np.digitize(data_points, bins)
print('DIGITIZED: ', digitized)
pdf = [len(list(filter(lambda x: x == b, digitized))) / total_count for b in bins]

x = np.arange(1, bs, 1)
chi2 = stats.chi2.pdf(x, df=1)
# chi2[0] = 0.0

print(pdf)
print(chi2)

# https://www.statology.org/kl-divergence-python/
# #define multiple Chi-square distributions
# plt.plot(x, stats.chi2.pdf(x, df=2), label='df: 2')
ret = scipy.special.rel_entr(pdf, chi2)
print(ret)
print(sum(ret))

print(sum(scipy.special.kl_div(pdf, chi2)))

#################################################################################### KL - END



