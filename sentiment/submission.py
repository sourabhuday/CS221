#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    wordsCount = dict()
    for word in x.split():
        if word not in wordsCount:
            wordsCount[word] = 1
        else:
            wordsCount[word] += 1
    return wordsCount
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and
      validationExamples to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def predict(x):
        phi = featureExtractor(x)
        if dotProduct(weights,phi)<0.0:
            return -1
        else:
            return 1
        
    for i in range(numEpochs):
        for item in trainExamples:
            x,y = item
            phi = featureExtractor(x)
            hinge = dotProduct(phi, weights) * y
            if hinge < 1: 
                increment(weights, -eta*-y, phi)
        print(i, evaluatePredictor(trainExamples, predict), evaluatePredictor(validationExamples, predict))
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified
      correctly by |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # note that there is intentionally flexibility in how you define phi.
    # y should be 1 or -1 as classified by the weight vector.
    # IMPORTANT: In the case that the score is 0, y should be set to 1.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        for item in random.sample(list(weights), random.randint(1, len(weights))):
            phi[item] = random.randint(1, 100)
        if dotProduct(weights, phi) == 0:
            y = 0
        elif dotProduct(weights, phi) > 0:
            y = 1
        else:
            y = -1
        # END_YOUR_CODE
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that 1 <= n <= len(x).
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        # raise Exception("Not implemented yet")
        features = dict()
        x = x.replace(' ', '')
        for i in range(0, len(x) - (n-1)):
            if x[i:i+n] in features:
                features[x[i:i+n]] += 1
            else:
                features[x[i:i+n]] = 1
        return features 
        # END_YOUR_CODE
    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)

    mu = []
    z = [0] * (len(examples))
    def newMuFeature(newMu, example):
        for feature in example:
            if feature in newMu:
                newMu[feature] += example[feature]
            else:
                newMu[feature] = example[feature]
        return newMu
    
    def lossFunc(example, centroid):
        sum = 0
        for feature in example:
            if feature in centroid:
                sum += (example[feature] - centroid[feature])**2
            else:
                sum += (example[feature])**2
        return sum

    for i in range(K):
        centroid = examples[random.randint(0,len(examples))]
        mu.append(centroid)

    for p in range(maxEpochs):
        for i, example in enumerate(examples):
            min = 99999
            for j, centroid in enumerate(mu):
                loss = lossFunc(example, centroid)
                if loss < min:
                    min = loss
                    z[i] = j
        prevMu = mu.copy()
        for i, centroid in enumerate(mu):
            newMu = {}
            pointCount = 0
            for j, example in enumerate(examples):
                if z[j] == i:
                    newMu = newMuFeature(newMu, example)
                    pointCount += 1
            if pointCount != 0:
                for feature in newMu:
                    newMu[feature] /= pointCount
            mu[i] = newMu

        if prevMu == mu:
            loss = 0
            for i, example in enumerate(examples):
                centroid = mu[z[i]]
                loss += lossFunc(example, centroid)
            return (mu, z, loss)
        
    loss = 0
    for i, example in enumerate(examples):
        centroid = mu[z[i]]
        loss += lossFunc(example, centroid)

    return (mu, z, loss)
    # END_YOUR_CODE
