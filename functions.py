import numpy as np
import pandas as pd
import random
import nltk
from collections import defaultdict
from IPython.display import display
from sklearn.model_selection import train_test_split


# util for calcutaling the emission (likelihood) prob
def emissionProb(word, tag, trainingSet):
    noOfTags = 0
    countWordGivenTag = 0
    for tupple in trainingSet:
        if tupple[1] == tag:
            noOfTags += 1
            if tupple[0] == word:
                countWordGivenTag += 1

    return countWordGivenTag / noOfTags


# util for calcutaling the transition prob
def transitionProb(tag2, tag1, trainingSet):
    countOfTag1 = 0
    countOfTag2GivenTag1 = 0
    trainSetLength = len(trainingSet)
    for i in range(trainSetLength):
        tupple = trainingSet[i]
        if tupple[1] == tag1:
            countOfTag1 += 1
            if i < trainSetLength-1 and trainingSet[i+1][1] == tag2:
                countOfTag2GivenTag1 += 1

    return countOfTag2GivenTag1 / countOfTag1


# util for transition prob matrix
def tagMatrix(trainingSet, tags):
    matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i in range(len(tags)):
        tag1 = tags[i]
        for j in range(len(tags)):
            tag2 = tags[j]
            matrix[i, j] = transitionProb(tag2, tag1, trainingSet)
    return matrix


# implementation of viterbi algorithm
def viterbi(words, trainingSet, tagList, transProbMatrix):
    state = []
    taggedWordsList = []
    tagsDict = defaultdict(list)
    for i in range(len(tagList)):
        tagsDict[tagList[i]].append(i)

    for i in range(len(words)):
        word = words[i]
        probabilities = []
        for tag in tagList:
            if i == 0:
                transitionProbability = transProbMatrix[tagsDict['.']
                                                        [0]][tagsDict[tag][0]]
            else:
                transitionProbability = transProbMatrix[tagsDict[state[-1]]
                                                        [0]][tagsDict[tag][0]]
            emmissionProbability = emissionProb(word, tag, trainingSet)
            probability = emmissionProbability * transitionProbability
            probabilities.append(probability)
        maxIndex = -1
        viterbiValue = -1
        for i in range(len(probabilities)):
            if probabilities[i] > viterbiValue:
                viterbiValue = probabilities[i]
                maxIndex = i
        maxState = tagList[maxIndex]
        state.append(maxState)
        wordAndTag = (word, maxState)
        taggedWordsList.append(wordAndTag)
    return taggedWordsList
