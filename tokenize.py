#/----------IMPORT SECTION----------/
import math
import torch
import sys
import random
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
from collections import Counter

#/----------CLASS DEFINITIONS SECTION----------/

#//VOCABULARY CLASS

class BisVocabulary():
  
  def __init__(self, 
               path_X: str,
               window_size: int = 1,
               stride: int = 1,
              ):
    """
      BisDataset constructor, inizialize a dataset for BIS tagging problem

      :param path_X: path where instances are stored
      :param window_size: size of window for text slicing
      :param stride: stride in slicing

    """
    #//Parameters
    self.window_size = window_size
    self.stride = stride
    self.unknown_token = "<UNK>"
    self.padding_token = "<PAD>"

    #//Dictionaries 
    self.word2occur_dictionary = Counter() #it's composed as follows {key: word, value: number of word's occurencies}
    self.word2index_dictionary = {}        #{key: word, value: unique index assigned to the word} 
    self.index2word_dictionary = {}       #{key:unique index assigned to the word, value: word }
    self.BIS2class_dictionary = {}        #{key: BIS sequence, value: class} where class is an integer in [0, number of possible classes]

    #//Operations
    self.word2occur_dictionary[self.unknown_token] = 1
    self.word2occur_dictionary[self.padding_token] = 1

    self.build(path_X)

    self.word2index_dictionary = self.word2index_repr()
    self.index2word_dictionary = self.index2word_repr()
    self.BIS2class_dictionary = {"B":0, "I":1, "S":2}


  def __len__(self):
    return(len(self.word2occur_dictionary))

  def window_slicing(self,
                     l_X: str
                     ):
    """
    :param l_X: instace string 

    :returns: a tuple which contains ("instance","target")

    Example: ("hello","BIII") 
    this happens if window_size = 4
    """
    return [(l_X[i:i+(self.window_size)]) for i in range(0,len(l_X),self.stride) if i + (self.window_size) < len(l_X)]
    
  def organize_data(self, 
                    l_X: str
                    ):
    """
      This method organizes data in a specific form 

      :param l_X: raw text paragraph from file

      :returns: A tuple composed as follows ([K_1,K_2,...,K_n])
    """
    l_X = l_X.rstrip('\n')

      
    return self.window_slicing(l_X.lower())


  def index2word_repr(self):
    """
      This method returns a dictionary which contains as key the word index and as value the word with
      the given index. 
      word2index_dictionary = ["a" : 1 , "b" : 2 ,"ciao" : 3 ,...]
      index2word_dictionary = [ 1 : "a", 2 : "b" , 3 : "ciao",...]
      It perform key, value swapping from word2index dictionary 
    """
    return {v:k for k,v in self.word2index_dictionary.items()}


  def word2index_repr(self):
    """
      This method returns the index encoding for the dictionary using the format
      dictionary       = ["a" : B , "b" : B ,"ciao" : BIII ,...]
      index_dictionary = ["a" : 1 , "b" : 2 ,"ciao" : 3    ,...]
      each token has its index number, using this encoding it's possible to 
      build the one hot encoding vector repr where each word is a vector 
      filled with zeros except for its own index position filled with 1
    """
    return {k:index for index,(k,_) in enumerate(self.word2occur_dictionary.items())}
    
  def build(self, 
                path_X: str
                ):
    """
      Read data from a file in path. 

      :param path_X: path where instances are stored
      :param path_y: path where targets are stored

      :returs: dictionary {K: word_i, V: bis_sequence_i}
    """
    dataset = {}
    with open(path_X) as X:
      for l_X in X:
        for t in self.organize_data(l_X):
          self.word2occur_dictionary.update({t : 1})


#//CONVERTER CLASS

class converter():
  
  def __init__(self,
               vocabulary: BisVocabulary, 
               model,
	       device,
               path_X: str,
               token_length: int,
               window_size: int = 1,
               stride: int = 1,
               padding: int = 1,
               max_size: int = math.inf
              ):
    """
      BisDataset constructor, inizialize a dataset for BIS tagging problem

      :param path_X: path where instances are stored
      :param path_y: path where targets are stored
      :param window_size: size of window for text slicing
      :param stride: stride in slicing

    """
    #//Parameters
    self.window_size = window_size
    self.stride = stride
    self.padding = padding
    self.vocabulary = vocabulary
    self.token_length = token_length
    self.model = model
    self.device = device
    self.class2BIS = {0:"B", 1:"I", 2:"S"}

    #//Lists
    self.token_list_line = []  #it contains a list of lists. In each list there are tokens in a given line
    self.dataset = []
    
    #//Operation
    self.read_data(path_X)
    self.dataset = self.window_slicing(self.token_list_line)
    self.token_list_line = []


  def transform(self):
    with torch.no_grad():
      len_vocab = len(self.vocabulary)
      window_len = (self.window_size * 2)+1
      dimension_1 = len_vocab * window_len
      pred = ""
      for line in self.dataset:
                  inputs = torch.stack(line)
                  one_hot_input = torch.zeros((inputs.shape[0], dimension_1))
                  for i, x in enumerate(inputs):
                    for j,e in enumerate(x):
                            one_hot_input[i, e+(j*len_vocab)] = 1
                  output = self.model(one_hot_input.to(self.device))
                  softout = torch.softmax(output,1)
                  prediction = torch.argmax(softout,dim=1)
                  for i in prediction.tolist():
                    pred+=self.class2BIS[i] 
                 
                  pred+='\n'
    return pred


  def window_slicing(self,
                     tokens: list
                     ):
    """
    :returns: a tuple which contains ("instance","target")

    Example: ("hello","BIII") 
    this happens if window_size = 4
    """
    d = []
    buffer = []
    for line_X in tokens:
        d = []
        for i in range(self.window_size,len(line_X)-self.window_size):
            key = torch.tensor([self.vocabulary.word2index_dictionary[element] 
                   if element in self.vocabulary.word2index_dictionary 
                   else self.vocabulary.word2index_dictionary["<UNK>"] 
                   for element in line_X[i-self.window_size:(i+self.window_size)+1]])            
            d.append(key)
        buffer.append(d)
    return buffer

  def tokenize(self,
              X: str):
        X = X.rstrip('\n')
        padding_list = ["<PAD>"] * self.window_size
        X_token = [X[i:i+self.token_length] if i+self.token_length <= len(X) else X[i:len(X)]+("$"*(self.token_length - (len(X)-i))) for i in range(0,len(X),self.token_length)]
        return (padding_list + X_token + padding_list)

  def read_data(self, 
                path_X: str, 
                ):
    """
      Read data from a file in path. 

      :param path_X: path where instances are stored
      :param path_y: path where targets are stored
    """
    with open(path_X) as X:
      for l_X in X:
        self.token_list_line.append(self.tokenize(l_X.lower()))


#//DATASET CLASS

"""
The idea behind this class is to build an infrastructure to provide data to the model.
The aim is to obtain a list which is composed as follows:

[([index(w_{i-window_size}),...,index(w_{i}),...,index(w_{i+window_size}]), [index(target(w_{i}))]),...]


Example:

window_size = 1  window looks like this -> [-1|current|+1]

 [23,2,100],  [9]
 [54,324,2],  [4]
 [13,76,12],  [8]
 [3,22,310],  [1]
"""

class BisDataset(torch.utils.data.Dataset):
  
  def __init__(self,
               vocabulary: BisVocabulary, 
               path_X: str,
               path_y: str,
               token_length: int,
               window_size: int = 1,
               stride: int = 1,
               padding: int = 1,
               max_size: int = math.inf
              ):
    """
      BisDataset constructor, inizialize a dataset for BIS tagging problem

      :param path_X: path where instances are stored
      :param path_y: path where targets are stored
      :param window_size: size of window for text slicing
      :param stride: stride in slicing

    """
    #//Parameters
    self.window_size = window_size
    self.stride = stride
    self.padding = padding
    self.vocabulary = vocabulary
    self.token_length = token_length
    
    #//Lists
    self.token_list_line = []  #it contains a list of lists. In each list there are tokens in a given line
    self.dataset = []
    
    #//Operation
    self.read_data(path_X,path_y)
    self.dataset = self.window_slicing(self.token_list_line)
    self.token_list_line = []
    self.dataset_max_size = min(max_size,len(self.dataset))

  def __len__(self):
    return(self.dataset_max_size)
    
  def __getitem__(self,index):
    return (self.dataset[index][0],self.dataset[index][1])

  def window_slicing(self,
                     tokens: list
                     ):
    """
    :returns: a tuple which contains ("instance","target")

    Example: ("hello","BIII") 
    this happens if window_size = 4
    """
    d = []
    for line_X,line_y in tokens:
        for i in range(self.window_size,len(line_X)-self.window_size):
            key = torch.tensor([self.vocabulary.word2index_dictionary[element] 
                   if element in self.vocabulary.word2index_dictionary 
                   else self.vocabulary.word2index_dictionary["<UNK>"] 
                   for element in line_X[i-self.window_size:(i+self.window_size)+1]])
            value = torch.tensor([self.vocabulary.BIS2class_dictionary[line_y[i]] 
                                  if line_y[i] in self.vocabulary.BIS2class_dictionary 
                                  else self.vocabulary.BIS2class_dictionary["I"]])
            d.append((key,value))
    random.shuffle(d)
    return d

  def tokenize(self,
              X: str,
              y: str
              ):
        X = X.rstrip('\n')
        y = y.rstrip('\n')
        padding_list = ["<PAD>"] * self.window_size
        X_token = [X[i:i+self.token_length] if i+self.token_length <= len(X) else X[i:len(X)]+("$"*(self.token_length - (len(X)-i))) for i in range(0,len(X),self.token_length)]
        y_token = [y[i:i+self.token_length] if i+self.token_length <= len(y) else y[i:len(y)]+("$"*(self.token_length - (len(y)-i))) for i in range(0,len(y),self.token_length)]
        
        return (padding_list + X_token + padding_list ,padding_list + y_token + padding_list)


    
  def read_data(self, 
                path_X: str, 
                path_y: str
                ):
    """
      Read data from a file in path. 

      :param path_X: path where instances are stored
      :param path_y: path where targets are stored
    """
    with open(path_X) as X, open(path_y) as y:
      for l_X, l_y in zip(X,y):
        self.token_list_line.append(self.tokenize(l_X.lower(),l_y))

#//TRAINER CLASS

class Trainer():

    def __init__(self, model, optimizer, device, loss, vocabulary_size):
        self.device = device
        self.loss_function = loss
        self.model = model
        self.optimizer = optimizer
        self.vocabulary_size = vocabulary_size
        self.model.to(self.device) 

    def tensor2ohe(self,sample, dimension_1):
          inputs = sample[0]
          targets = sample[1].reshape((sample[1].shape[0]),)
          one_hot_input = torch.zeros(((sample[0].shape[0]), dimension_1))
          for i, x in enumerate(inputs):
            for j, e in enumerate(x):
                one_hot_input[i, e+(j*self.vocabulary_size)] = 1
          return one_hot_input,targets

    def train(self, train_dataset, output_folder, epochs=1):
        self.model.train()
        window_len = (dset.window_size * 2)+1
        dimension_1 = self.vocabulary_size * window_len
        for epoch in range(epochs):
            for sample in train_dataset:
                inputs, targets = self.tensor2ohe(sample,dimension_1)
                output_distribution = self.model(inputs.to(self.device))
                loss = self.loss_function(output_distribution, targets.to(self.device))  
                loss.backward()  
                self.optimizer.step()
                self.optimizer.zero_grad()


#/----------PARAMETERS SECTION----------/
BASE_PATH = sys.argv[1]
vocabulary_window_size = 1
dataset_window_size = 3
stride = 1
token_length = 1
batch_size = 64

#/----------VOCABULARY AND TRAINING DATASET DEFINITION----------/

X = BASE_PATH+".sentences.train"
y = BASE_PATH+".gold.train"

vocabulary  = BisVocabulary(X, window_size=vocabulary_window_size,stride=stride)

dset = BisDataset(vocabulary,
                  X,
                  y,
                  window_size=dataset_window_size,
                  token_length=token_length,
		  max_size=1500000)

dl = torch.utils.data.DataLoader(dset,batch_size=batch_size)


#/----------MODEL SECTION----------/

#//Model parameters
embedding_dim = 300
class_number = 3 if "merged" not in BASE_PATH else 2
dimension_0 = len(vocabulary)*((dataset_window_size * 2)+1) #It represent the one-hot-encoding input vector

#//Model definition
model = nn.Sequential(nn.Linear(dimension_0, embedding_dim ),
		      nn.Sigmoid(),
		      nn.Linear(embedding_dim, 30 ),
 		      nn.Sigmoid(),
		      nn.Linear(30, class_number ))

#/----------TRAINER SECTION----------/

#//Trainer parameters
device = 'cpu'
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.00009,0.000099))
loss = F.cross_entropy

#//Trainer definition
trainer = Trainer(model, optimizer, device, loss, len(vocabulary))

#/----------TRAIN MODEL----------/

trainer.train(dl, BASE_PATH, epochs=6)

#/----------CONVERT TEXT----------/

convert = BASE_PATH+".sentences.test"
c = converter(vocabulary,model,device,convert,token_length=token_length,window_size=dataset_window_size)
print(c.transform())





