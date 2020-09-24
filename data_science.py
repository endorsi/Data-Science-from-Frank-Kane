# Data Science from Frank Kane
# functions represent the topics

import scipy.stats as sp
from sklearn.metrics import r2_score

#from pylab import *

import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

def one():

    # PROBABILITY GRAPHS

    from scipy.stats import norm
    from scipy.stats import expon
    from scipy.stats import binom
    from scipy.stats import poisson

    x = np.arange(-3,3,0.01)

    plt.plot(x,norm.pdf(x))
    # pdf = probability density function for continuous values
    # norm means normal, you can also put expon there
    y = np.arange(0,10,0.01)
    plt.plot(y,binom.pmf(y,10,0.5))
    plt.show()

    # my website gets on avarage 500 visitors per day, what is the odds of getting 550 ?
    mu = 500
    z=np.arange(300,600,2)
    plt.plot(z,poisson.pmf(z,mu))
    plt.show()

def two():

    # HISTOGRAM and MOMENTS

    vals = np.random.normal(0,1,10000)
    plt.hist(vals,50) # 50 = number of columns

    plt.savefig('C:\\Users\\Samsung\\Desktop\\myplot.png',format='png')
    plt.show()

    print(np.percentile(vals,50))
    print(np.percentile(vals,10))

    print(np.mean(vals)) #first moment
    print(np.var(vals))  #second moment-variance
    print(sp.skew(vals)) #third moment-skew (shows which side the data is centered)
    print(sp.kurtosis(vals)) #fourth moment-kurosis (which describes the shape of the tail, for a normal distrubition this is 0)

def three():

    # BASIC PLOTTING

    x = np.arange(-3, 3, 0.01)
    axes = plt.axes()
    axes.set_xlim([-5, 5])
    axes.set_ylim([0, 1.0])
    axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    plt.plot(x, norm.pdf(x),'g-') # green solid line
    plt.plot(x, norm.pdf(x, -1, 0.5),'r:') # -1 specifies center, 0.5 multiplies by two
                           # red color with hashes
                            # other keywords ( --  -. )

    plt.legend(["A","B"],loc=4)
    plt.show()

def four():

    # CARTOON AND PIE

    plt.xkcd()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    ax.set_ylim([-30, 10])
    data = np.ones(100)
    data[70:] -= np.arange(30)
    plt.annotate(
        "AAAAAAAA",
        xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

    plt.plot(data)
    plt.xlabel('zaman')
    plt.ylabel('ruh sağlığım')

    plt.show()

    # PIE CHART

    plt.rcdefaults() # remove previous mode

    values = [12, 55, 4, 32, 14]
    colors = ['r', 'g', 'b', 'c', 'm']
    explode = [0, 0, 0.2, 0, 0]
    labels = ['India', 'United States', 'Russia', 'China', 'Europe']
    plt.pie(values, colors= colors, labels=labels, explode = explode)
    plt.title('Student Locations')
    plt.show()

def five():

    # BARS , SCATTER , CORRELATION

    values = [12, 55, 4, 32, 14]
    colors = ['r', 'g', 'b', 'c', 'm']
    plt.bar(range(0,5), values, color= colors)
    plt.show()

    X = np.random.normal(3.0, 1.0, 1000)
    Y = np.random.normal(50.0, 10.0, 1000)

    plt.scatter(X,Y)
    plt.show()

    print(np.corrcoef(X,Y))
    # gives an array of correlation
    # -0.00569 is the number we ara looking for
    # since it is close to zero, there is no corr. at all


def six():

    # LINEAR REGRESSION

    X = np.random.normal(3.0, 1.0, 1000) # (center , distrubition , total dots)
    Y = 100 - (X + np.random.normal(0, 0.3, 1000))
    plt.scatter(X,Y)

    slope, intercept, r_value, p_value, std_err = sp.linregress(X, Y)

    print(r_value**2) #r-square
    # close to 1, which is great, almost all of the variance is captured

    fitLine = slope * X + intercept
    plt.plot(X, fitLine, c='r')
    plt.show()

def seven():

    # POLYNOMIAL REGRESSION

    np.random.seed(2) # for getting same random numbers each time
    # i selected 5

    pageSpeeds = np.random.normal(3.0, 1.0, 1000)
    purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
    plt.scatter(pageSpeeds, purchaseAmount)
    plt.show()

    x = np.array(pageSpeeds)
    y = np.array(purchaseAmount)
    p3 = np.poly1d(np.polyfit(x, y, 3)) # third degree polynomial
    # you can change degree but there should not be overfitting
    # the line shouldnt capture outliners and should be lower degree as possible

    xp = np.linspace(0, 7, 100) # from 0 to 7 on x axis
    plt.scatter(x, y)
    plt.plot(xp, p3(xp), c='r')
    plt.show()

    r2 = r2_score(y, p3(x))
    print(r2)
    #0.78 is fine since it is close to 1

def eight():

    #MULTIVARIATE REGRESSION

    df = pd.read_excel('C:\\Users\\Samsung\\Desktop\\cars.xls')
    print(df.head())

    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()

    X = df[['Mileage', 'Cylinder', 'Doors']]
    y = df['Price']

    X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())
    print (X)
    est = sm.OLS(y, X).fit() #Ordinary Least Squares
    print(est.summary())
    #Cylinder has the highest coef. which means it affects the price moslty
    #Price = a + b1*mileage + b2*cylinder + b3*doors

    print(y.groupby(df.Doors).mean())
    # there is not much difference between mean prices

def nine():

    # TRAIN - TEST (POLYNOMIAL REGRESSION)

    np.random.seed(2)

    pageSpeeds = np.random.normal(3.0, 1.0, 100)
    purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds
    plt.scatter(pageSpeeds, purchaseAmount)
    plt.show()

    #spliting train test data
    trainX = pageSpeeds[:80]#80 and everything before 80
    testX = pageSpeeds[80:]#everything after 80

    trainY = purchaseAmount[:80]
    testY = purchaseAmount[80:]

    plt.scatter(testX, testY)
    plt.show()

    x = np.array(trainX)
    y = np.array(trainY)

    for i in range(1,9): #from 1st degree polynomial to eight

        p4 = np.poly1d(np.polyfit(x, y, i))

        xp = np.linspace(0, 7, 100)
        axes = plt.axes()
        axes.set_xlim([0,7])
        axes.set_ylim([0, 200])
        plt.scatter(x, y)
        plt.plot(xp, p4(xp), c='r')
        plt.show()

        r2 = r2_score(testY, p4(testX))
        print(i , " = " , r2)
        # 6th degree seems best (%60)

def ten():

    # ML Naive Bayesian Method

    import os
    import io
    import numpy
    from pandas import DataFrame
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    def readFiles(path):
        for root, dirnames, filenames in os.walk(path):
           for filename in filenames:
                path = os.path.join(root, filename)

                inBody = False
                lines = []
                f = io.open(path, 'r', encoding='latin1')
                for line in f:
                    if inBody:
                        lines.append(line)
                    elif line == '\\n':
                        inBody = True
                f.close()
                message = '\\n'.join(lines)
                yield path, message

    def dataFrameFromDirectory(path, classification):
        rows = []
        index = []
        for filename, message in readFiles(path):
            rows.append({'message': message, 'class': classification})
            index.append(filename)

        return DataFrame(rows, index=index)

    data = DataFrame({'message': [], 'class': []})

    data = data.append(dataFrameFromDirectory('C:\\Users\Samsung\Desktop\DataScience-Python3\emails\spam', 'spam'))
    data = data.append(dataFrameFromDirectory('C:\\Users\\Samsung\\Desktop\\DataScience-Python3\\emails\\ham', 'ham'))

    print(data.head())

    vectorizer = CountVectorizer()
    # count the number of times that word occurs

    counts = vectorizer.fit_transform(data['message'].values)
    #converts words into numbers in sparse matrix as a numerical index into an array

    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)

    examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]

    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print(predictions)

def eleven():

    #K-Means Clustering

    def createClusteredData(N, k):
        random.seed(10)
        pointsPerCluster = float(N)/k
        X = []
        for i in range (k):
            incomeCentroid = random.uniform(20000.0, 200000.0)
            ageCentroid = random.uniform(20.0, 70.0)
            for j in range(int(pointsPerCluster)):
                X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
        X = np.array(X)
        return X

    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale
    from numpy import random

    data = createClusteredData(100, 5)

    for i in range(1,9): # K means from 1 to 8

        model = KMeans(n_clusters=i)

        # Note I'm scaling the data to normalize it! Important for good results
        # for seeing a square area
        model = model.fit(scale(data))

        # We can look at the clusters each data point was assigned to"
        print(model.labels_)

        # And we'll visualize it:
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(np.float))
        # c specifies colors
        plt.show()

        # as you see, the K number should be 4


def twelve():

    # ML Decision Trees - Random Forest

    from sklearn import tree
    input_file = r"C:\Users\Samsung\Desktop\DataScience-Python3\PastHires.csv"
    df = pd.read_csv(input_file, header = 0)
    print(df.head())

    #scikit-learn needs everything to be numerical for decision trees to work. So, we'll map Y,N to 1,0 and levels of education to some scale of 0-2. In the real world, you'd need to think about how to deal with unexpected or missing data! By using map(), we know we'll get NaN for unexpected values.

    # converts Y to 1 N to 0 for making them numerical
    d = {'Y': 1, 'N': 0}
    df['Hired'] = df['Hired'].map(d)
    df['Employed?'] = df['Employed?'].map(d)
    df['Top-tier school'] = df['Top-tier school'].map(d)
    df['Interned'] = df['Interned'].map(d)
    d = {'BS': 0, 'MS': 1, 'PhD': 2}
    df['Level of Education'] = df['Level of Education'].map(d)
    print(df.head())

    #Next we need to separate the features from the target column that we're trying to bulid a decision tree for

    features = list(df.columns[:6]) # taking first 6 columns as a feature

    #training
    y = df["Hired"]
    X = df[features]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,y)

    from IPython.display import Image,display
    from sklearn.externals.six import StringIO
    import pydotplus

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                             feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    # save as an png
    graph.write_png("tree.png")
    # if the statement is true, the flowchart goes to direction left, otherwise right
    # gini gives entropy value, 0 means all of them is same
    # value represents [ "number of people not hired" , "hired" ]
    # the flowchart kewps going until it reach entropy zero

    from sklearn.ensemble import RandomForestClassifier

    for i in range(5,16):
      for testing in range(5):
        clf = RandomForestClassifier(n_estimators=i)
        # number of tree i want to predict
        clf = clf.fit(X, y)
        #Predict employment of an employed 10-year veteran\n",
        print (i, "\n" ,clf.predict([[10, 1, 4, 0, 0, 0]]))
        #...and an unemployed 10-year veteran"
        print (clf.predict([[10, 0, 4, 0, 0, 0]]))

        #it gives different results depending on your number of trees
        #it also doesnt give same results when testing

def thirteen():

    # Support Vector Machines (SVM)


    # Create fake income/age clusters for N people in k clusters\n",
    def createClusteredData(N, k):
        pointsPerCluster = float(N)/k
        X = []
        y = []
        for i in range (k):
            incomeCentroid = np.random.uniform(20000.0, 200000.0)
            ageCentroid = np.random.uniform(20.0, 70.0)
            for j in range(int(pointsPerCluster)):
               X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
               y.append(i)
        X = np.array(X)
        y = np.array(y)
        return X, y


    (X, y) = createClusteredData(100, 5)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()

    from sklearn import svm, datasets

    C = 1.0 # error penalty term default value
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    # some other kernel types: RBF, poly..

    def plotPredictions(clf):

        xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                         np.arange(10, 70, 0.5))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        plt.figure(figsize=(8, 6))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
        plt.show()

        # it divided the groups for predicting new values

    plotPredictions(svc)

    print(svc.predict([[200000, 40]]))

def fourteen():

    # USER BASED COLLABORATIVE SYSTEM - PANDAS

    r_cols = ['user_id', 'movie_id', 'rating']
    ratings = pd.read_csv('C:\\Users\Samsung\Desktop\DataScience-Python3\ml-100k\\u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

    m_cols = ['movie_id', 'title']
    movies = pd.read_csv('C:\\Users\Samsung\Desktop\DataScience-Python3\ml-100k\\u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

    ratings = pd.merge(movies, ratings)

    print("\nXXXXXXXXXXXXXX 1 XXXXXXXXXXXXX\n",)
    print(ratings.head()) # first 5 rows

    movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
    # creating table with rows as user_id columns as titles and values as rating

    print("\nXXXXXXXXXXXXXX 2 XXXXXXXXXXXXX\n",)
    print(movieRatings.head())

    starWarsRatings = movieRatings['Star Wars (1977)']

    print("\nXXXXXXXXXXXXXX 3 XXXXXXXXXXXXX\n",)
    print(starWarsRatings.head())

    # Pandas corrwith function makes it really easy to compute the pairwise correlation of Star Wars' vector of user rating with every other movie! After that, we'll drop any results that have no data, and construct a new DataFrame of movies and their correlation score (similarity) to Star Wars
    similarMoviestoSW = movieRatings.corrwith(starWarsRatings)

    similarMoviestoSW = similarMoviestoSW.dropna()
    # dropping NaN's (Not a Number)

    df = pd.DataFrame(similarMoviestoSW)
    print("\nXXXXXXXXXXXXXX 4 XXXXXXXXXXXXX\n",)
    print(df.head(10))

    print("\nXXXXXXXXXXXXXX 5 XXXXXXXXXXXXX\n",)
    print(similarMoviestoSW.sort_values(ascending=False))
    # Hallow reed, Man of the year ? Something is wrong
    # The movies which rated by few people should be removed

    movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
    print("\nXXXXXXXXXXXXXX 6 XXXXXXXXXXXXX\n",)
    print(movieStats.head())
    # 12 angry men is rated by 125 people, the number 125 could be selected for limiting
    # Let's get rid of any movies rated by fewer than 125 people, and check the top-rated ones that are left

    popularMovies = movieStats['rating']['size'] >= 125

    print("\nXXXXXXXXXXXXXX 7 XXXXXXXXXXXXX\n",)
    print(movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]) #first 15 rows

    #Similar Movies to Star Wars Table, Similarity Column is joined to our Movie Stats which are rated by more than 125 people
    df = movieStats[popularMovies].join(pd.DataFrame(similarMoviestoSW, columns=['similarity']))
    print("\nXXXXXXXXXXXXXX 8 XXXXXXXXXXXXX\n",)
    print(df.head())

    print("\nXXXXXXXXXXXXXX 9 XXXXXXXXXXXXX\n",)
    print(df.sort_values(['similarity'], ascending=False)[:15])#first 15 rows
    #Better results (Star wars should be removed since it is the same movie)

def fifteen():

    # ITEM BASED RECOMMEND SYSTEM

    r_cols = ['user_id', 'movie_id', 'rating']
    ratings = pd.read_csv('C:\\Users\Samsung\Desktop\DataScience-Python3\ml-100k\\u.data', sep='\t', names=r_cols,
                          usecols=range(3), encoding="ISO-8859-1")

    m_cols = ['movie_id', 'title']
    movies = pd.read_csv('C:\\Users\Samsung\Desktop\DataScience-Python3\ml-100k\\u.item', sep='|', names=m_cols,
                         usecols=range(2), encoding="ISO-8859-1")

    ratings = pd.merge(movies, ratings)
    print(ratings.head())  # first 5 rows

    userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
    print(userRatings.head())

    #creating corr matrix with comparing each movie
    corrMatrix = userRatings.corr()
    print(corrMatrix.head())

    #However, we want to avoid spurious results that happened from just a handful of users that happened to rate the same pair of movies. In order to restrict our results to movies that lots of people rated together
    #and also give us more popular results that are more easily recongnizable - we'll use the min_periods argument to throw out results where fewer than 125 users rated a given movie pair

    #corr method = pearson, rated by at least 125 people
    corrMatrix = userRatings.corr(method='pearson', min_periods=125)
    print(corrMatrix.head())

    #Now let's produce some movie recommendations for user ID 0, who added to the data set as a test case. This guy really likes Star Wars and The Empire Strikes Back, but hated Gone with the Wind.
    # extract his ratings from the userRatings DataFrame, and use dropna() to get rid of missing data

    myRatings = userRatings.loc[0].dropna()
    print(myRatings)

    simCandidates = pd.Series()

    for i in range(0, len(myRatings.index)):
        print("Adding sims for " + myRatings.index[i] + "...")

        # Retrieve similar movies to this one that I rated
        sims = corrMatrix[myRatings.index[i]].dropna()

        # Now scale its similarity by how well I rated this movie
        sims = sims.map(lambda x: x * myRatings[i])

        # Add the score to the list of similarity candidates
        simCandidates = simCandidates.append(sims)

    #Glance at our results so far
    print("sorting...")
    simCandidates.sort_values(inplace = True, ascending = False)
    print(simCandidates.head(10))

    # This is starting to look like something useful! Note that some of the same movies came up more than once, because they were similar to more than one movie I rated.
    # We'll use groupby() to add together the scores from movies that show up more than once, so they'll count more
    simCandidates = simCandidates.groupby(simCandidates.index).sum()

    simCandidates.sort_values(inplace=True, ascending=False)
    print(simCandidates.head(10))

    # filtering the movies i rated
    filteredSims = simCandidates.drop(myRatings.index)
    print(filteredSims.head(10))

    # for better results:
    # i can penalize to movies which is similar to the movies user rated 1
    # i can throw away to users which rated too much movies


def sixteen():

    # KNN

    # the purpose is finding the rating of a movie
    # by the distances found between genres and popularity scores

    r_cols = ['user_id', 'movie_id', 'rating']
    ratings = pd.read_csv('C:\\Users\Samsung\Desktop\DataScience-Python3\ml-100k\\u.data', sep='\t', names=r_cols,
                          usecols=range(3), encoding="ISO-8859-1")
    print(ratings.head())

    movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
    print(movieProperties.head())


    # The raw number of ratings isn't very useful for computing distances between movies
    # so we'll create a new DataFrame that contains the normalized number of ratings.
    # So, a value of 0 means nobody rated it, and a value of 1 will mean it's the most popular movie there is

    movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
    movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    print(movieNormalizedNumRatings.head())

    # we'll put together everything into one big Python dictionary called movieDict.
    # Each entry will contain the movie name, list of genre values, the normalized popularity score, and the average rating for each movie:

    movieDict = {}

    with open("C:\\Users\Samsung\Desktop\DataScience-Python3\ml-100k\\u.item", encoding = "ISO-8859-1") as f:
        temp = ''
        for line in f:
            #line.decode("ISO-8859-1")
            fields = line.rstrip('\n').split('|')
            movieID = int(fields[0]) #extracting movie ID
            name = fields[1]    #movie name
            genres = fields[5:25] #movie genres (total 19)
            genres = map(int, genres)
            movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'),
                                  movieProperties.loc[movieID].rating.get('mean'))

    print(movieDict[1])

    # Now let's define a function that computes the "distance" between two movies based on how similar their genres are,
    # and how similar their popularity is. Just to make sure it works, we'll compute the distance between movie ID's 2 and 4

    from scipy import spatial

    def ComputeDistance(a, b):
        genresA = a[1]
        genresB = b[1]
        # finding the distance between genres (cosine since its a array)
        genreDistance = spatial.distance.cosine(genresA, genresB)
        popularityA = a[2]
        popularityB = b[2]
        # this time we only subtract since the values are float numbers
        popularityDistance = abs(popularityA - popularityB)
        return genreDistance + popularityDistance

    print(ComputeDistance(movieDict[2], movieDict[4]))

    # Now, we just need a little code to compute the distance between some given test movie (Toy Story, in this example) and all of the movies in our data set.
    # When the sort those by distance, and print out the K nearest neighbors

    import operator

    # finding the distances for the given movie comparing to each movie and sorting them
    def getNeighbors(movieID, K):
        distances = []
        for movie in movieDict:
            if (movie != movieID):
                dist = ComputeDistance(movieDict[movieID], movieDict[movie])
                distances.append((movie, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(K):
            neighbors.append(distances[x][0])
        return neighbors

    # adding the ratings of neighbours we found and divide the by the value K
    K = 10
    avgRating = 0
    neighbors = getNeighbors(1, K)
    for neighbor in neighbors:
        avgRating += movieDict[neighbor][3]
        print (movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))

    avgRating /= K

    print(avgRating) # estimated rating
    print(movieDict[1][3]) # real rating
    # not bad, pretty close


def seventeen():

    # DIMENSIONALITY REDUCTION WITH PCA AND SVD

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    import pylab as pl
    from itertools import cycle

    iris = load_iris()

    print(iris)

    numSamples, numFeatures = iris.data.shape
    print(numSamples),
    print(numFeatures)
    print(list(iris.target_names))
    # as you see there are 4 features which makes it 4D
    # we are gonna convert it to 2D

    print(type(iris.data))
    X = iris.data
    #whiten does normalization
    pca = PCA(n_components=2, whiten=True).fit(X)
    # 2 represents 2D
    X_pca = pca.transform(X)

    print(pca.components_)

    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    #That's pretty cool. Although we have thrown away two of our four dimensions, PCA has chosen the remaining two dimensions well enough that we've captured 92% of the variance in our data in a single dimension alone!
    #The second dimension just gives us an additional 5%; altogether we've only really lost less than 3% of the variance in our data by projecting it down to two dimensions

    colors = cycle('rgb')
    target_ids = range(len(iris.target_names))
    pl.figure()
    for i, c, label in zip(target_ids, colors, iris.target_names):
        pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1],
            c=c, label=label)
    pl.legend()
    pl.show()

def eighteen():

    # SVM WITH K-FOLD CROSS VALIDATION


    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn import datasets
    from sklearn import svm
    iris = datasets.load_iris()

    # Split the iris data into train/test data sets with 40% reserved for testing
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

    # Build an SVC model for predicting iris classifications using training data
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

    # Now measure its performance with the test data\n",
    print(clf.score(X_test, y_test))

    # We give cross_val_score a model, the entire data set and its "real" values, and the number of folds
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)

    # Print the accuracy for each fold:
    print(scores)

    # And the mean accuracy of all 5 folds:
    print(scores.mean())

    # Trying Polynomial
    clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores)
    print(scores.mean())
    # getting the same score, since there could be over-fitting in poly, then the linear is a better choice

    # Build an SVC model for predicting iris classifications using training data
    clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
    # Now measure its performance with the test data
    print(clf.score(X_test, y_test))
    # the score we get is higher, which means if we didnt use K-fold we would think that polynomial kernel model is better

def nineteen():

    # OUTLIERS

    incomes = np.random.normal(500, 10000, 100000) # average 500 with derivation 1000, total 100,000 value
    incomes = np.append(incomes, [100000000])  # adding extremely rich
    incomes = np.append(incomes, [1000000])
    incomes = np.append(incomes, [10000])
    incomes = np.append(incomes, [100]) # adding extremely poor
    incomes = np.append(incomes, [1])

    import matplotlib.pyplot as plt
    plt.hist(incomes, 50)
    plt.show()
    # it looks awful because of outliers, lets pretend that we should throw them out

    print(incomes.mean()) # 3x higher than it should be

    def reject_outliers(data):
        u = np.median(data)
        s = np.std(data) # calculates standart deviation
        # filtered the values higher than
        # (2*Standart Deviation + median value)
        filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
        return filtered

    filtered = reject_outliers(incomes)
    plt.hist(filtered, 50)
    plt.show()

    print(np.mean(filtered))
    # almost same the value i gave as an average (500)

def twenty():

    # A/B TEST, T-TEST, P-VALUES

    from scipy import stats
    # A represents our test data
    A = np.random.normal(25.5, 10.0, 10000)
    # B reprenests our control data (the way it used to be)
    B = np.random.normal(26.0, 10.0, 10000)

    print(stats.ttest_ind(A, B))
    # the t-value is negative, we should stop the test

    # same test with 10x bigger data
    A = np.random.normal(25.5, 10.0, 100000)
    B = np.random.normal(26.0, 10.0, 100000)

    print(stats.ttest_ind(A, B))


    # same test with 100x bigger data
    A = np.random.normal(25.5, 10.0, 1000000)
    B = np.random.normal(26.0, 10.0, 1000000)

    print(stats.ttest_ind(A, B))

    # from results the general form does not change so much with bigger data
    # you should test it and decide how long you should run your test

    print(stats.ttest_ind(A, A))
    # t-value is zero because there is no difference

    A = np.random.normal(27.0, 10.0, 10000)
    B = np.random.normal(26.0, 10.0, 10000)
    print(stats.ttest_ind(A, B))
    # the p-value is lower than %1 which is great
    # also the t-value is positive which means test has a positive impact


def twentyone():

    # TENSORFLOW MOST BASIC PROBLEM

    import tensorflow as tf
    a = tf.Variable(1, name="a")
    b = tf.Variable(2, name="b")
    f = a + b

    init = tf.global_variables_initializer()
    with tf.Session() as s:
        init.run()
        print( f.eval() )

def twentytwo():

    # TENSORFLOW MNIST

    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    def display_sample(num):

        #Print this sample's label
        print(Y_train[num])
        label = Y_train[num]
        #Reshape the 768 values to a 28x28 image
        image = X_train[num].reshape([28,28])
        plt.title('Sample: %d  Label: %d' % (num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()



    display_sample(423)


def twentythree():

    # TENSORFLOW - KERAS , MNIST

    from tensorflow import keras
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import RMSprop

    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

    # We need to explicitly convert the data into the format Keras / TensorFlow expects.
    # We divide the image data by 255 in order to normalize it into 0-1 range, after converting it into floating point values

    train_images = mnist_train_images.reshape(60000, 784)
    # There are total 60K train and 10K test images
    test_images = mnist_test_images.reshape(10000, 784)
    #28 x 28 = total 784 pixels, we made them 1D array
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255

    #Now we'll convert the 0-9 labels into "one-hot" format, i.e [0,0,0,0,0,0,0,1,0,0]

    train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
    test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

    #Let's take a peek at one of the training images just to make sure it looks OK

    def display_sample(num):

        #Print the one-hot array of this sample's label
        print(train_labels[num])
        #Print the label converted back to a number
        label = train_labels[num].argmax(axis=0) # axis 0 represents rows, 1 represents columns
        #Reshape the 768 values to a 28x28 image
        image = train_images[num].reshape([28,28])
        plt.title('Sample: %d  Label: %d' % (num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()

    display_sample(1534)

    # We can set up the same layers like this. The input layer of 784 features feeds into a ReLU layer of 512 nodes,
    # which then goes into 10 nodes with softmax applied.

    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy']) # calculate accuracy at each epoch

    history = model.fit(train_images, train_labels,

                        batch_size=100,
                        epochs=10,
                        verbose=2, # will mention the number of epoch, "0" will show you nothing, 1 will show you [=======]
                        validation_data=(test_images, test_labels))


    # to see wrong predicted numbers
    for x in range(1000,10000):

        test_image = test_images[x,:].reshape(1,784)
        predicted_cat = model.predict(test_image).argmax()
        label = test_labels[x].argmax()
        if (predicted_cat != label):
            plt.title('Prediction: %d Label: %d' % (predicted_cat, label))
            plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
            plt.show()


def twentyfour():

    # TENSORFLOW - BINARY PROBLEM

    # PREDICTING POLITICAL PARTY BASED ON VOTES (DEMOCRAT OR REPUBLICIAN)

    feature_names =  ['party','handicapped-infants', 'water-project-cost-sharing',
                        'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                        'el-salvador-aid', 'religious-groups-in-schools',
                        'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                        'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                        'education-spending', 'superfund-right-to-sue', 'crime',
                        'duty-free-exports', 'export-administration-act-south-africa']

    voting_data = pd.read_csv('D:\\Udemy\Data Science from Frank Kane\DataScience-Python3\house-votes-84.data.txt',
                              na_values=['?'],
                              names = feature_names)
    print(voting_data.head())

    print(voting_data.describe())
    # We can see there's some missing data to deal with here; some politicians abstained on some votes, or just weren't present when the vote was taken.
    # We will just drop the rows with missing data to keep it simple, but in practice you'd want to first make sure that doing so didn't introduce any sort of bias into your analysis
    # (if one party abstains more than another, that could be problematic for example.)

    voting_data.dropna(inplace=True) # inplace=True means it will overwrite the data
                                    # otherwise it will create just a copy
    print(voting_data.describe())

    voting_data.replace(('y', 'n'), (1, 0), inplace=True)
    voting_data.replace(('democrat', 'republican'), (1, 0), inplace=True)


    all_features = voting_data[feature_names].drop('party', axis=1).values
    print(type(all_features))
    print(all_features)
    print(all_features[0])
    all_classes = voting_data['party'].values

    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.models import Sequential
    from sklearn.model_selection import cross_val_score

    def create_model():
        model = Sequential()
        #16 feature inputs (votes) going into an 32-unit layer
        model.add(Dense(32, input_dim=16, kernel_initializer='normal', activation='relu'))
        # Another hidden layer of 16 units
        model.add(Dense(16, kernel_initializer='normal', activation='relu'))
        # Output layer with a binary classification (Democrat or Republican political party)
        # sigmoid (2-element softmax) is useful for binary problems
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

    # Wrap our Keras model in an estimator compatible with scikit_learn
    estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
    # Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
    cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
    print(cv_scores.mean())
    # %94 without even trying too hard !
twentyfour()

def twentyfive():

    # TENSORFLOW - CNN

    import tensorflow

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import RMSprop

    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

    # We need to shape the data differently then before.
    # Since we're treating the data as 2D images of 28x28 pixels instead of a flattened stream of 784 pixels,
    # we need to shape it accordingly. Depending on the data format Keras is set up for, this may be 1x28x28 or 28x28x1
    # (the "1" indicates a single color channel, as this is just grayscale.
    # If we were dealing with color images, it would be 3 instead of 1 since we'd have red, green, and blue color channels



    from tensorflow.keras import backend as K

    if K.image_data_format() == 'channels_first':
        train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255
    test_images /= 255

    # we need to convert our train and test labels to be categorical in one-hot format

    train_labels = tensorflow.keras.utils.to_categorical(mnist_train_labels, 10)
    test_labels = tensorflow.keras.utils.to_categorical(mnist_test_labels, 10)

    # Setting up a convolutional neural network involves more layers.Not all of these are strictly necessary;
    # you could run without pooling and dropout, but those extra steps help avoid overfitting and help things run faster.\n",

    # We'll start with a 2D convolution of the image - it's set up to take 32 windows, or "filters",
    # of each image, each filter being 3x3 in size.

    # We then run a second convolution on top of that with 64 3x3 windows - this topology is just what comes recommended within Keras's own examples.
    # Again you want to re-use previous research whenever possible while tuning CNN's, as it is hard to do.

    # Next we apply a MaxPooling2D layer that takes the maximum of each 2x2 result to distill the results down into something more manageable.

    # A dropout filter is then applied to prevent overfitting.

    # Next we flatten the 2D layer we have at this stage into a 1D layer.
    # So at this point we can just pretend we have a traditional multi-layer perceptron..

    # and feed that into a hidden, flat layer of 128 units.\n",

    # We then apply dropout again to further prevent overfitting.

    #And finally, we feed that into our final 10 units where softmax is applied to choose our category of 0-9.

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    # 64 3x3 kernels
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Reduce by taking the max of each 2x2 block
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout to avoid overfitting
    model.add(Dropout(0.25))
    # Flatten the results to one dimension for passing into our final layer
    model.add(Flatten())
    # A hidden layer to learn with
    model.add(Dense(128, activation='relu'))
    # Another dropout
    model.add(Dropout(0.5))
    # Final categorization from 0-9 with softmax
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels,
                        batch_size=32,
                        epochs=1,
                        verbose=2,
                        validation_data=(test_images, test_labels))

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # %98.23 accuracy with just one epoch

def twentysix():

    # TENSORFLOW - RNN

    from tensorflow.keras.preprocessing import sequence
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.datasets import imdb

    # Now import our training and testing data.
    # We specify that we only care about the 20,000 most popular words in the dataset
    # in order to keep things somewhat managable.

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=20000)

    print(x_train[0])
    # That doesn't look like a movie review! But this data set has spared you a lot of trouble -
    # they have already converted words to integer-based indices.
    # The actual letters that make up a word don't really matter as far as our model is concerned, what matters are the words themselves
    # and our model needs numbers to work with, not letters.

    # So just keep in mind that each number in the training features represent some specific word.

    print(y_train[0])
    # They are just 0 or 1, which indicates whether the reviewer said they liked the movie or not.

    # RNN's can blow up quickly, so again to keep things managable
    # on our little PC let's limit the reviews to their first 50 words
    # that is a way of truncating our back-propagation through time
    x_train = sequence.pad_sequences(x_train, maxlen=50)
    x_test = sequence.pad_sequences(x_test, maxlen=50)

    # We will start with an Embedding layer - this is just a step that converts the input data
    # into dense vectors of fixed size that's better suited for a neural network.
    # You generally see this in conjunction with index-based text data like we have here.
    # The 20,000 indicates the vocabulary size (remember we said we only wanted the top 20,000 words)
    # and 128 is the output dimension of 128 units.

    # Next we just have to set up a LSTM layer for the RNN itself. It's that easy.
    # We specify 128 to match the output size of the Embedding layer,
    # and dropout terms to avoid overfitting, which RNN's are particularly prone to.

    # Finally we just need to boil it down to a single neuron with a sigmoid
    # activation function to choose our binay sentiment classification of 0 or 1

    model = Sequential()
    model.add(Embedding(20000, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=5,
              verbose=2,
              validation_data=(x_test, y_test))

    # let's evaluate our model's accuracy:

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=32,
                                verbose=2)
    print('Test score:', score)
    print('Test accuracy:', acc)

def twentyseven():

    # FINAL REVIEW WITH KNN, SVM, LR

    masses_data = pd.read_csv('C:\\Users\Samsung\Desktop\DeepLearning\mammographic_masses.data.txt')
    print(masses_data.head())

    # Make sure you use the optional parmaters in read_csv to convert missing data (indicated by a ?) into NaN,
    # and to add the appropriate column names (BI_RADS, age, shape, margin, density, and severity)
    masses_data = pd.read_csv('C:\\Users\Samsung\Desktop\DeepLearning\mammographic_masses.data.txt',
                              na_values=['?'],
                              names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
    print(masses_data.head())

    print(masses_data.describe())

    # There are quite a few missing values in the data set. Before we just drop every row that's missing data,
    # let's make sure we don't bias our data in doing so. Does there appear to be any sort of correlation to what sort of data has missing fields?
    # If there were, we'd have to try and go back and fill that data in.
    print( masses_data.loc[(masses_data['age'].isnull()) |
                  (masses_data['shape'].isnull())   |
                  (masses_data['margin'].isnull())  |
                  (masses_data['density'].isnull())] )

    # data seems randomly distributed, go ahead and drop rows with missing data
    masses_data.dropna(inplace=True)
    print(masses_data.describe())


    all_features = masses_data[['age', 'shape',
                                 'margin', 'density']].values
    all_classes = masses_data['severity'].values

    feature_names = ['age', 'shape', 'margin', 'density']

    print(all_features)

    # normalizing the data
    from sklearn import preprocessing

    scaler = preprocessing.StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    print(all_features_scaled)


    #KNN
    from sklearn import neighbors
    from sklearn.model_selection import cross_val_score

    for n in range(1,50):
        clf =  neighbors.KNeighborsClassifier(n_neighbors=n)
        cv_scores= cross_val_score(clf,all_features_scaled,all_classes)
        print(n,cv_scores.mean())

    # K=7 seems like the best option

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes)
    print("LR = ",cv_scores.mean())

    # SVM with rbf
    from sklearn import svm

    C = 1.0
    svc = svm.SVC(kernel = "rbf" , C=C)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes)
    print("SVM = ", cv_scores.mean())

