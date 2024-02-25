import nltk
from nltk.stem.snowball import FrenchStemmer
from RulesList import RulesList
import random
import numpy
import tflearn
import tensorflow as tf
import os
import datetime

class Chatbot:
    ERROR_THRESHOLD = 0.7

    def __init__(self, ignoreWords : list = ['?', '!'], approvalWords : list = ['y', 'o', 'yes', 'oui'], verbose : bool = False, forceSave : bool = False, learn : bool = True, update : bool = True):
        self.roots          = []
        self.ruleList       = []
        self.corpus         = []
        self.ignoreWords    = ignoreWords
        self.approvalWords  = approvalWords
        self.verbose        = verbose
        self.forceSave      = forceSave
        self.learn          = learn
        self.update         = update
        self.stemmer        = FrenchStemmer()
        self.rules          = None
        self.model          = None

    def readRules(self, filename : str) -> None:
        self.rules = RulesList('rules.json')
        self.rules.readRules()

    def preprocessing(self) -> None:
        for rule in self.rules.getRule():
            if self.verbose:
                print('[preprocessing] Traitement de la règle', rule)
            ruleName = rule.getRuleName()
            for pattern in rule.getPatterns():
                word = nltk.word_tokenize(pattern)
                self.roots.extend(word)
                self.corpus.append((word, ruleName))
            if ruleName not in self.ruleList:
                self.ruleList.append(ruleName)
        self.ruleList = sorted(self.ruleList)
        self.roots = [self.stemmer.stem(w.lower()) for w in self.roots if w not in self.ignoreWords]
        self.roots = sorted(list(set(self.roots)))
        if self.verbose:
            print('Corpus de ', len(self.corpus), ' phrases')
            print('Repartition en ', len(self.ruleList), ' classes')
            print(len(self.roots), ' racines uniques : ', self.roots)

    def trainData(self) -> None:
        training = []
        ouput = []
        outputEmpty = [0] * len(self.ruleList)

        for doc in self.corpus:
            group = []
            patterns = doc[0]
            patterns = [self.stemmer.stem(word.lower()) for word in patterns]
            for word in self.roots:
                group.append(1) if word in patterns else group.append(0)
            outputRow = list(outputEmpty)
            outputRow[self.ruleList.index(doc[1])] = 1
            training.append((group, outputRow))

        random.shuffle(training)
        train_x = [entry[0] for entry in training]
        train_y = [entry[1] for entry in training]

        nn = tflearn.input_data(shape = [None, len(train_x[0])])
        nn = tflearn.fully_connected(nn, 8)
        nn = tflearn.fully_connected(nn, 8)
        nn = tflearn.fully_connected(nn, len(train_y[0]), activation = 'softmax')
        nn = tflearn.regression(nn)
        self.model = tflearn.DNN(nn, tensorboard_dir = 'logs')
        if os.path.isfile('model.tflearn.index') and not self.forceSave:
            self.model.load('./model.tflearn')
            if self.verbose:
                print('[trainData] Modèle chargé depuis ./model.tflearn')
            return None
        self.model.fit(train_x, train_y, n_epoch = 1000, batch_size = 8, show_metric = True)
        if os.path.isfile('model.tflearn.index') or self.forceSave:
            self.model.save('./model.tflearn')
            if self.verbose:
                print('[trainData] Modèle enregistré dans ./model.tflearn')

    def tokenize(self, sentence : str) -> list:
        words = nltk.word_tokenize(sentence)
        words = [self.stemmer.stem(word.lower()) for word in words]
        if self.verbose:
            print('[tokenize] Tokens : ', words)
        return words

    def searchGroup(self, sentence) -> numpy.array:
        tokens = self.tokenize(sentence)
        group = [0] * len(self.roots)
        if self.verbose:
            print('[searchGroup] Racines : ', self.roots)
        for tok in tokens:
            for i, word in  enumerate(self.roots):
                if word == tok:
                    group[i] = 1
                    if self.verbose:
                        print('[searchGroup] Occurence trouvée : ', word)
        if self.verbose:
            print('[searchGroup] Groupe : ', group)
        return (numpy.array(group))

    def classification(self, sentence : str) -> list:
        groups = self.searchGroup(sentence)
        if not 1 in groups:
            return False
        results = self.model.predict([groups])[0]
        results = [[i, res] for i, res in enumerate(results) if res > Chatbot.ERROR_THRESHOLD]
        results.sort(key = lambda x : x[1], reverse = True)
        resultList = []
        for res in results:
            resultList.append((self.ruleList[res[0]], res[1]))
        return resultList

    def askForResponse(self, sentence : str) ->str:
        print(self.complete(random.choice(self.rules.getQuestion())))
        ruleName = ''
        patterns = [sentence]
        responses = []
        sentence = input('> ')
        if sentence in self.approvalWords:
            print('Veuillez saisir le nom de la règle puis tapez sur entrée.')
            ruleName = input('> ')
            while not ruleName or any(ruleName == rule.getRuleName() for rule in self.rules.getRule()):
                print('Nom invalide ou déjà utilisé, veuillez recommencer.')
                sentence = input('> ')
            print('Saisissez les variantes de la question de départ en retournant à la ligne entre chacune d\'entre elles. Lorsque vous avez terminé ou si vous ne souhaitez pas en ajouter, tapez "." puis entrée.')
            sentence = input('> ')
            while sentence != '.':
                patterns.append(sentence)
                sentence = input('> ')
            print('Saisissez les réponses attendues en retournant à la ligne entre chacune d\'entre elles. Lorsque vous avez terminé, tapez "." puis entrée.')
            sentence = input('> ')
            while sentence != '.':
                responses.append(sentence)
                sentence = input('> ')
            self.rules.addRule(ruleName, patterns, responses, self.update)
            if self.update:
                self.forceSave = True
                chatbot.preprocessing()
                chatbot.trainData()
                return 'Règle ajoutée aux données d\'entrainement, modèle mis à jour'
            return 'Règle ajoutée aux données d\'entrainement, modèle non à jour'
        return 'Puis-je vous aider autrement?'



    def response(self, sentence : str) ->str:
        results = self.classification(sentence)
        if self.verbose:
            print('[response] ', results)
        if results:
            while results:
                for rule in self.rules.getRule():
                    if rule.getRuleName() == results[0][0]:
                        return random.choice(rule.getResponses())
                results.pop(0)
        if self.learn:
            return self.askForResponse(sentence)
        return random.choice(self.rules.getUnknown())

    def complete(self, sentence) ->str:
        date = datetime.datetime.now()
        sentence = sentence.replace('{hour}', str(date.hour))
        sentence = sentence.replace('{mn}', str(date.minute))
        sentence = sentence.replace('{sec}', str(date.second))
        return sentence

    def interact(self, ) -> None:
        while True:
            sentence = input('> ')
            if sentence == '.':
                break
            print(self.complete(self.response(sentence)))

if __name__ == '__main__':
    chatbot = Chatbot(verbose = True)
    chatbot.readRules('rules.json')
    chatbot.preprocessing()
    chatbot.trainData()
    chatbot.interact()