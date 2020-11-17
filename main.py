from functions import *

# calculating the accuracy of the viterbi algorithm


def calcAccuracy(taggedSequence, testTaggedWordsList):
    correctTagCount = 0
    # length of taggedSequence and testTaggedWords is same
    for i in range(len(taggedSequence)):
        if taggedSequence[i] == testTaggedWordsList[i]:
            correctTagCount += 1
    return correctTagCount/len(taggedSequence)


print('---------------------------------------------------------------------------------------------------------------------')
print('Downloading the required the tagset')
# download the treebank corpus from nltk
nltk.download('treebank')

# download the universal tagset from nltk
nltk.download('universal_tagset')

data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))

print('Downloading complete')
print('---------------------------------------------------------------------------------------------------------------------')

# training set : testing set :: 8:2
trainingSet, testingSet = train_test_split(
    data, train_size=0.80, test_size=0.20, random_state=101)

# creating the list of training and testing tagged words
trainingTaggedWordsList = []
for sentence in trainingSet:
    for tupple in sentence:
        trainingTaggedWordsList.append(tupple)

testingTaggedWordsList = []
for sentence in testingSet:
    for tupple in sentence:
        testingTaggedWordsList.append(tupple)

print('No of tagged words in training set: ', len(trainingTaggedWordsList))
print('No of tagged words in testing set: ', len(testingTaggedWordsList))


uniqueTags = set()
for tupple in trainingTaggedWordsList:
    uniqueTags.add(tupple[1])

print('No of unique tags: ', len(uniqueTags))
print('Tags:', uniqueTags)
print('---------------------------------------------------------------------------------------------------------------------')

# getting the transition prob matrix
uniqueTagsList = list(uniqueTags)
transitionProbMatrix = tagMatrix(
    trainingTaggedWordsList, uniqueTagsList)
tagsDataFrame = pd.DataFrame(
    transitionProbMatrix, uniqueTagsList, uniqueTagsList)
display(tagsDataFrame)
print('---------------------------------------------------------------------------------------------------------------------')

# generating 10 random numbers
# we use seed here so that if same random sentences are used when this code is executed multiple time and we get some consistent results
random.seed(1000)
randomNoList = []
for i in range(10):
    randomNoList.append(random.randint(i, len(testingSet)))

# getting 10 random sentences from testing set
testingSentences = []
for i in randomNoList:
    testingSentences.append(testingSet[i])

# getting the list of words and corresponding tags from randomly generated test set
testTaggedWordsList = []
for sentence in testingSentences:
    for tupple in sentence:
        testTaggedWordsList.append(tupple)

# getting the list of untagged words from randomly generated test set
testUntaggedWordsList = []
for sentence in testingSentences:
    for tupple in sentence:
        testUntaggedWordsList.append(tupple[0])

print('Testing 10 sentences')
# getting the tag sequence for randomly generated testing set
taggedSequence = viterbi(testUntaggedWordsList,
                         trainingTaggedWordsList, uniqueTagsList, transitionProbMatrix)

accuracy = calcAccuracy(taggedSequence, testTaggedWordsList)
print('Viterbi Algorithm Accuracy: ', accuracy * 100)
print('---------------------------------------------------------------------------------------------------------------------')

# Code to test all the test sentences
# This takes lot of time
# print('Testing all the sentences: ')
# testTaggedWordsList = []
# for sentence in testingSet:
#     for tupple in sentence:
#         testTaggedWordsList.append(tupple)

# # getting the list of untagged words from randomly generated test set
# testUntaggedWordsList = []
# for sentence in testingSet:
#     for tupple in sentence:
#         testUntaggedWordsList.append(tupple[0])

# taggedSequence = viterbi(testUntaggedWordsList, trainingTaggedWordsList, transitionProbMatrix)

# accuracy = calcAccuracy(taggedSequence, testTaggedWordsList)
# print('Viterbi Algorithm Accuracy: ', accuracy * 100)

# taking input sentences from user to tag it
print('Tagging input sentences based on the trained model:')
while(True):
    sentence = input('Enter the sentence: ')
    taggedSentence = viterbi(
        sentence.split(), trainingTaggedWordsList, uniqueTagsList, transitionProbMatrix)
    print(taggedSentence)
    i = input('Do you want to continue (y/n): ')
    if i == 'n' or i == 'N':
        break
