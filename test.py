import re
import unicodedata as ud
import codecs
import numpy as np
from collections import defaultdict
import io
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from pyvi import ViTokenizer
import re
import unicodedata as ud

file_name = ["./data/easy_clean.txt", "./data/hard_clean.txt"]
train = io.open(file_name[0], encoding="utf-8").readlines()

tagsetDict = {"N": 0,
              "V": 1,
              "A": 2,
              "P": 3,
              "M": 4,
              "D": 5,
              "R": 6,
              "E": 7,
              "C": 8,
              "I": 9,
              "X": 10}

inverseTagsetDict = {tagsetDict[k]: k for k in tagsetDict}
wordBank = defaultdict()
bigramBank = defaultdict()
bigramDict = defaultdict(int)

bigramFreq = {x: [0]*18 for x in tagsetDict}
mostCommonBigrams = defaultdict()

with codecs.open("./data/easy_clean.txt", "r", "UTF-8") as f:
    lines = f.readlines()
    for line in lines:
        line_split = line.split()
        for i, w in enumerate(line_split):
            parts = w.split("/")
            if i >= 1:
                prevParts = line_split[i-1].split("/")
            if len(parts) == 1 or parts[1] not in tagsetDict:
                continue

            word = parts[0]
            pos = parts[1]
            if i >= 1:
                prevWord = prevParts[0]
                prevPos = prevParts[1]
                bigramBank[word] = (pos, prevWord, prevPos)
                bigramDict[(pos, prevPos)] += 1

            if word not in wordBank:
                wordBank[word] = [pos]
            else:
                wordBank[word] += [pos]
bi_grams = []
tri_grams = []

for sentence in set(wordBank):
    temp = 0
    for s in list(sentence):
        if s == "_":
            temp += 1
    if temp == 1:
        bi_grams.append(sentence)
    elif temp == 2:
        tri_grams.append(sentence)
# print(bi_grams)
# print(tri_grams)

# SVM Classification
# finished getting training data
for k in bigramFreq:
    maxFreq = 0
    maxPos = "N"
    for i, x in enumerate(bigramFreq[k]):
        if x > maxFreq:
            maxFreq = x
            maxPos = inverseTagsetDict[i]
    mostCommonBigrams[k] = maxPos


def Viterbi_rule_based(word, wordIdx, lineSize, line):
    feat = [1]
    sentPercent = float(wordIdx)/float(lineSize)
    feat.append(sentPercent)

    if word[0].isupper() and wordIdx != 0:
        feat.append(1)
    else:
        feat.append(0)

    posIdx_array = ([0] * len(tagsetDict))
    posSet = []
    if word in wordBank:
        posSet = wordBank[word]
    else:
        if wordIdx == 0:
            posSet = list(tagsetDict.keys())[0]
            posIdx_array[tagsetDict[posSet]] = 1
            return feat + posIdx_array + [0]
        else:
            prevWord = line[wordIdx-1]
            if prevWord in wordBank:
                prevPos = wordBank[prevWord]
                maxPos = mostCommonBigrams[prevPos]
                if prevPos == "E":
                    maxPos = 3
                posIdx_array[tagsetDict[maxPos]] = 1
                feat += posIdx_array + [tagsetDict[maxPos]]
                return feat
            else:
                posSet = list(tagsetDict.keys())[0]
                posIdx_array[tagsetDict[posSet]] = 1
                feat += posIdx_array + [0]
                return feat

    for pos in posSet:
        posIdx = tagsetDict[pos]
        posIdx_array[posIdx] += 1.0 / len(wordBank[word])
    feat += (posIdx_array) + [0]
    return feat


y = []
X_train = []
for line in train:
    l_split = line.split()
    for i, w in enumerate(l_split):
        parts = w.split("/")
        word = parts[0]
        len_line = len(l_split)

        if len(parts) == 1 or parts[1] not in tagsetDict:
            continue
        y.append(wordBank[word][0])
        X_train.append(Viterbi_rule_based(word, i, len_line, l_split))

# print(len(X_train))
# print(len(y))
# for i in range(10):
#     print(i, ":", X_train[i])
# print(y[:10])

train_fit = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y)


def syllablize(sentence):
    word = '\w+'
    non_word = '[^\w\s]'
    digits = '\d+([\.,_]\d+)+'

    patterns = []
    patterns.extend([word, non_word, digits])
    patterns = f"({'|'.join(patterns)})"

    sentence = ud.normalize('NFC', sentence)
    tokens = re.findall(patterns, sentence, re.UNICODE)
    return [token[0] for token in tokens]


def longest_matching(sentence, bi_grams, tri_grams):
    syllables = syllablize(sentence)
    syl_len = len(syllables)
    curr_id = 0
    word_list = []
    done = False

    while (curr_id < syl_len) and (not done):
        curr_word = syllables[curr_id]
        if curr_id >= syl_len - 1:
            word_list.append(curr_word)
            done = True
        else:
            next_word = syllables[curr_id + 1]
            pair_word = ' '.join([curr_word.lower(), next_word.lower()])
            if curr_id >= (syl_len - 2):
                if pair_word in bi_grams:
                    word_list.append('_'.join([curr_word, next_word]))
                    curr_id += 2
                else:
                    word_list.append(curr_word)
                    curr_id += 1
            else:
                next_next_word = syllables[curr_id + 2]
                triple_word = ' '.join([pair_word, next_next_word.lower()])
                if triple_word in tri_grams:
                    word_list.append(
                        '_'.join([curr_word, next_word, next_next_word]))
                    curr_id += 3
                elif pair_word in bi_grams:
                    word_list.append('_'.join([curr_word, next_word]))
                    curr_id += 2
                else:
                    word_list.append(curr_word)
                    curr_id += 1
    return word_list


def toString(wl):
    wl = longest_matching(wl, bi_grams, tri_grams)
    X = []
    A = []
    text = ""
    for i in set(wl):
        X.append(Viterbi_rule_based(i, 1, 1, wl))
        A = str(train_fit.predict(X))

    # print(A)
    for i in range(len(wl)):
        text += wl[i]
        text += '/'
        text += str(A[2])
        text += ' '
    return text


wr = "Vì nó rất đặc biệt nên tôi đã chú ý"
print(wr)
wl = ViTokenizer.tokenize(wr)
wl = wl.split()
for i in wl:
    print(toString(i), end='')
