import numpy as np
import pandas as pd
import re
import random
import pickle
from numpy.random import choice

class Generator:

    train_dict = dict()
    dotsdict = dict()
    unique_words = []
    commasdict = dict()
    wcount = 0
    uwcount = 0
    notAllowedEnds = ['is', 'was', 'i am', 'then', 'than', 'that', 'this', 'the', 'a', 'an', 'but', 'my', 'your', 'his', 'our', 'her', 'every', 'i', 'when', 'why', 'who', 'to', 'in']
    questionMarkingWords = ['why', 'what', 'who', 'which', 'how']
    def isWordEnd(self, s):
        for nae in self.notAllowedEnds:
            if s == nae:
                return False
        if s[:-2] == "na":
            return False
        return True


#---------------------------------------LOWER IS REPLACE_PMARKS-------------------------------------------

    
    def replace_pmarks(self, s):
        s = s.replace('!', '.').replace('?', '.').replace('\n', '.') #пока что единственный знак окончания предложения - точка 
        s = s.replace(':', ' ').replace(';', ' ').replace('(', ' ').replace(')', ' ') #эта строка заменяет все ненужные символы на пробелы
        s = s.replace('-', ' ').replace('"', ' ')
        s = s.replace("’m", "#am").replace("’ll", "#will").replace("’ve", '#have').replace("gon’", "gonna").replace("in’", 'ing') #форматирование слов (существительные с глаголом "to be" и т.п. считаются за одно слово)
        s = s.replace("won’t", 'will#not').replace("can’t", "cann’t").replace("n’t", '#not').replace("’s ", '#is ').replace("’re", "#are").replace("’", ' ')
        s = re.sub(' +', ' ', s) #удаление всех лишних пробелов, образовавшихся в процессе обработки текста
        return s


#---------------------------------------LOWER IS TEXT_TO_LIST-------------------------------------------


    def text_to_list(self, s): #эта функция
        s = s.lower() #приводит текст к нижнему регистру  
        s_list = s.split(' ') #и разбивает по словам,
        return s_list #возвращая список


#---------------------------------------LOWER IS IMPORT_TEXTS-------------------------------------------


    def import_texts(self, path):#предобработка текста для обучения и разбиение на слова
        xxx = open(path, "r").read().replace('\n', ' ') #прочитали текст из файла и избавились от пробелов
        xxx = self.replace_pmarks(xxx) #избавились от всех ненужных символов
        xlist = self.text_to_list(xxx)#превратили обработанный текст в список слов
        for i in range(len(xlist)):
            xlist[i] = xlist[i].replace("#", " ") #замена спец. символов на пробелы
        self.wcount = len(xlist) #общее количество слов в тексте 
        return xlist #функция возвращает обработанный список


#---------------------------------------LOWER IS FIT-------------------------------------------


    def fit(self, input_dir): #обучение модели 
        self.commasdict = dict() #словарь вида {"word1*word2":num_of_commas}, где в качестве ключа используется пара слов, разделенныз символом, а значение - кол-во запятых между такими словами в тексте (в данном порядке)
        tsamples = self.import_texts(input_dir)
        self.unique_words = list(set(tsamples)) #список всех слов (каждое слово из текста по одному разу; слова типа "did not" и "did" считаются за различные)
        self.uwcount = len(self.unique_words)#количество уникальных слов
        for i in range(self.uwcount):
            self.unique_words[i] = self.unique_words[i].replace(".", "")
            self.unique_words[i] = self.unique_words[i].replace(",", "")
        self.dotsdict = dict.fromkeys(self.unique_words, float(0)) #словарь вида {"word":num_of_dots}, где ключи - элементы tsamples, а значения - кол-во точек, которые стоят после соотв. ключа в тексте
        for i in range(0, self.wcount): #цикл подсчёта значений dotsdict и удаления точек из элементов tsamples
            if tsamples[i].find(".") >= 0:
                tsamples[i] = tsamples[i].replace(".", "")
                self.dotsdict[tsamples[i]] += 1
            if i > 0: #инициализация ключей commasdict
                self.commasdict[tsamples[i - 1].replace(",", "") + '*' + tsamples[i].replace(",", "")] = float(0)
        for i in range(0, self.wcount - 1): #цикл подсчёта значений commasdict
            if tsamples[i].find(",") >= 0:
                tsamples[i] = tsamples[i].replace(",", "")
                self.commasdict[tsamples[i] + '*' + tsamples[i + 1].replace(",", "")] += 1.0
                self.train_dict = dict.fromkeys(tsamples) #словарь вида {"word":{"nw_1":prob_1, "nw_2":prob_2,...}}, где "word" - элемент unique_words, которому соответствует ещё один словарь из слов, которые хотя бы один раз в тексте встречались сразу после "word" 
        gdict = dict.fromkeys(self.unique_words, 0) #словарь вида {"word":entry_count}, кол-во вхождений в текстедля каждого слова из unique_words
        for i in range(0, self.wcount): #инициализация train_dict
            self.train_dict[tsamples[i]] = {}
        for i in range(self.wcount - 1):
            self.train_dict[tsamples[i]][tsamples[i + 1]] = float(0)
        for i in range(self.wcount - 1):
            self.train_dict[tsamples[i]][tsamples[i + 1]] += 1.0
            gdict[tsamples[i]] += 1
        for i in range(self.wcount):
            if gdict[tsamples[i]] == -1:
                continue
            cnt = sum(self.train_dict[tsamples[i]].values())
            self.train_dict[tsamples[i]] = {k: v / cnt for k, v in self.train_dict[tsamples[i]].items()}
            self.dotsdict[tsamples[i]] /= float(gdict[tsamples[i]])
            gdict[tsamples[i]] = -1


#---------------------------------------LOWER IS GENERATE-------------------------------------------


    def generate(self, model, length, seed="rand"): #генерация текста длиной length слов с первым словом seed (по умолчанию выбирается случайное слово)
        random.seed()
        seed = seed.lower() #приведение первого слова к "нормальному" виду
        if (seed == "rand") or not (self.train_dict.get(seed)): 
            seed = self.unique_words[random.randint(0, self.uwcount - 1)] #выбор случайного слова, если seed не задан/не существует
        s = seed.title() #строка начинается с seed (с заглавной буквы)
        lstwrd = seed
        eskount = 0
        nxtdot = False #флаг того, что уже можно закончить предложение
        nwucf = False #next word upper case flag, следующее слово должно начаться с большой буквы
        nxtqmark = False
        for i in range(1, length):
            wrdmap = self.train_dict[lstwrd] #wrdmap - словарь возможных продолжений 
            nxtwrd = choice(list(wrdmap.keys()), 1, p=list(wrdmap.values()))[0] #случайный выбор следующего слова (учитывая вероятности)
            if nxtwrd in self.questionMarkingWords:
                nxtqmark = True
            while (i == length - 1) and not (self.isWordEnd(nxtwrd)) and not(eskount >= 100):
                eskount += 1
                nxtwrd = choice(list(wrdmap.keys()), 1, p=list(wrdmap.values()))[0] #перевыбор последнего слова до тех пор, пока не найдётся подходящее
            if (not nxtdot) and (random.randint(1, 10) == 3): #в среднем, каждые 10 итераций будет включаться флаг nxtdot, говорящий о том, что
                nxtdot = True #можно закончить предложение, когда попадётся нужное слово
            if not self.commasdict[lstwrd + '*' + nxtwrd]: 
                self.commasdict[lstwrd + '*' + nxtwrd] = self.dotsdict[lstwrd]
            if (nxtdot) and (self.isWordEnd(lstwrd)):
                nxtdot = False # сброс флага
                nwucf = True # следующее слово начать с большой буквы
                if nxtqmark:
                    s += '?'
                else:
                    s += '.'
                nxtqmark = False
            elif (i > 0) and (self.commasdict[lstwrd + '*' + nxtwrd] > 3):
                s += ','
            s += ' '
            if nxtwrd == 'i' or nxtwrd[0:2] == 'i ': #для красоты
                s = s + 'I' + nxtwrd[1:]
            elif nwucf:
                s = s + nxtwrd.title()
                nwucf = False
            else:
                s += nxtwrd
            lstwrd = nxtwrd
        print(s + '.')


#---------------------------------------LOWER IS CONSTRUCTOR-------------------------------------------

        
    def __init__(self, input_dir, model):
        self.fit(input_dir)


#******************************************************************************************************
#---------------------------------------LOWER IS MAIN MODULE-------------------------------------------
#******************************************************************************************************

s = input('Put "t" to train model or "g" to generate text (default "t"): ')
if s[0] == 'g':
    model = input('Enter path to the model: ')
    length = int(input('Enter length of the text: '))
    seed = input('Enter the first word: ')
    inpo = open(model, "rb")
    esk = pickle.load(inpo)
    print('***********************')
    esk.generate(model, length, seed)
else:
    inpdir = input('Enter input directory: ')
    model = input('Enter path to save model: ')
    esskeetit = Generator(inpdir, model)
    oup = open(model + 'generator.pkl', "wb")
    pickle.dump(esskeetit, oup, 2)
    print("Done!")
