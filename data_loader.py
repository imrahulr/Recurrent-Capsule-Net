import re
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
replace_numbers=re.compile(r'\d+',re.IGNORECASE)    
    
def clean_data(text):    
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+\-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def text_to_wordlist(text, 
                     remove_stopwords=False, 
                     stem_words=False):
    
    text = text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)
    text=special_character_removal.sub('',text)
    text=replace_numbers.sub('n',text)

    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    return(text)

def load_data(name, 
              path_data_dir, 
              max_len=150, 
              max_features=100000):
        
    train_df = pd.read_csv('{}/train.csv.zip'.format(path_data_dir))
    test_df = pd.read_csv('{}/test.csv.zip'.format(path_data_dir))
    
    if name=='toxic':
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        text_col = 'comment_text'
    
    if name=='agnews':
        list_classes = ['out_1', 'out_2', 'out_3', 'out_4']
        text_col = 'text'
        train_df[text_col] = train_df['title'] + train_df['des']
        test_df[text_col] = test_df['title'] + test_df['des']
        train_df['out'].astype('object', inplace=True)
        train_df = pd.concat([train_df, pd.get_dummies(train_df['out'], prefix='out')], axis=1)
        test_df['out'].astype('object', inplace=True)
        test_df = pd.concat([test_df, pd.get_dummies(test_df['out'], prefix='out')], axis=1)
        
    if name=='yelp_polarity':
        list_classes = ['out_1', 'out_2']
        text_col = 'text'
        train_df['out'].astype('object', inplace=True)
        train_df = pd.concat([train_df, pd.get_dummies(train_df['out'], prefix='out')], axis=1)
        test_df['out'].astype('object', inplace=True)
        test_df = pd.concat([test_df, pd.get_dummies(test_df['out'], prefix='out')], axis=1)
        
    if name=='yelp':
        text_col = 'text'
        list_classes = ['star_1', 'star_2', 'star_3', 'star_4', 'star_5']
        train_df['star'].astype('object', inplace=True)
        train_df = pd.concat([train_df, pd.get_dummies(train_df['star'], prefix='star')], axis=1)
        test_df['star'].astype('object', inplace=True)
        test_df = pd.concat([test_df, pd.get_dummies(test_df['star'], prefix='star')], axis=1)
       
    if name=='imdb':
        text_col = 'text'
        list_classes = ['output']
    
    print('Processing text dataset')
    train_df[text_col] = train_df[text_col].map(lambda x: clean_data(x))
    test_df[text_col] = test_df[text_col].map(lambda x: clean_data(x))

    list_sentences_train = train_df[text_col].fillna("NA").values
    y = train_df[list_classes].values
    list_sentences_test = test_df[text_col].fillna("NA").values

    comments = []
    for text in list_sentences_train:
        comments.append(text_to_wordlist(text))

    test_comments=[]
    for text in list_sentences_test:
        test_comments.append(text_to_wordlist(text))

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    train_data_pre = pad_sequences(sequences, maxlen=max_len)
    print('Shape of pre train_data tensor:', train_data_pre.shape)
    train_data_post = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    print('Shape of post train_data tensor:', train_data_post.shape)
    
    print('Shape of train_label tensor:', y.shape)
    
    test_data_pre = pad_sequences(test_sequences, maxlen=max_len)
    print('Shape of pre test_data tensor:', test_data_pre.shape)
    test_data_post = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    print('Shape of post test_data tensor:', test_data_post.shape)
    
    return word_index, train_data_pre, train_data_post, y, test_data_pre, test_data_post

def load_embeddings(embeddings_path, 
                    word_index, 
                    max_features=100000, 
                    embed_size=300):
    
    print('Loading word vectors')
    count = 0
    embeddings_index = {}
    
    f = open(embeddings_path)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
        coef = embeddings_index[word]
    f.close()
    
    print('Found %d word vectors of glove.' % len(embeddings_index))
    emb_mean,emb_std = coef.mean(), coef.std()
    print(emb_mean, emb_std)
    print('Total %s word vectors.' % len(embeddings_index))
    
    print('Preparing embedding matrix')
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix

def save_predictions(test_predicts, 
                     name, 
                     output_dir):
    if name=='toxic':
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        test_predicts = pd.DataFrame(data=test_predicts, columns=list_classes)
            
    if name=='agnews':
        list_classes = ['out_1', 'out_2', 'out_3', 'out_4']
        test_predicts = pd.DataFrame(data=test_predicts, columns=list_classes)
        test_predicts['out'] = test_predicts.idxmax(axis=1)
        test_predicts['out'] = test_predicts['out'].apply(lambda x: re.sub(r"[^0-9+]", "", x))
        test_predicts = test_predicts.drop(list_classes, axis=1)
        
    if name=='yelp_polarity':
        list_classes = ['out_1', 'out_2']
        test_predicts = pd.DataFrame(data=test_predicts, columns=list_classes)
        test_predicts['out'] = test_predicts.idxmax(axis=1)
        test_predicts['out'] = test_predicts['out'].apply(lambda x: re.sub(r"[^0-9+]", "", x))
        test_predicts = test_predicts.drop(list_classes, axis=1)
    
    if name=='yelp':
        list_classes = ['star_1', 'star_2', 'star_3', 'star_4', 'star_5']
        test_predicts = pd.DataFrame(data=test_predicts, columns=list_classes)
        test_predicts['star'] = test_predicts.idxmax(axis=1)
        test_predicts['star'] = test_predicts['star'].apply(lambda x: re.sub(r"[^0-9+]", "", x))
        test_predicts = test_predicts.drop(list_classes, axis=1)
    
    if name=='imdb':
        list_classes = ['output']
        test_predicts = pd.DataFrame(data=test_predicts, columns=list_classes)
    
    test_predicts.to_csv("{}/test_predictions.csv".format(output_dir), index=False)
    