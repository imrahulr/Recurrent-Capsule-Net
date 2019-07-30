import os
import argparse, sys 
import pandas as pd
import numpy as np

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from utils import RocAucEvaluation
from data_loader import load_data, load_embeddings, save_predictions
from models import CapsuleNetwork
 
def train(model, 
          data, 
          data_post, 
          y, 
          test_data, 
          test_data_post, 
          output_dir, 
          valid_split=0.1, 
          num_epochs=15, 
          batch_size=128):
    
    file_path = "{}/capsule_single.h5".format(output_dir)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    callbacks_list = [checkpoint, early] 
    hist = model.fit([data, data_post], y, epochs=num_epochs, batch_size=128, shuffle=True, validation_split=0.05, 
                     callbacks =callbacks_list, verbose=1)

    model.load_weights(file_path)
    test_predicts = model.predict([test_data, test_data_post], batch_size=1024, verbose=1)
    return test_predicts
    
def train_folds(model, 
                data, 
                data_post, 
                y, 
                test_data, 
                test_data_post, 
                output_dir, 
                fold_count=10, 
                num_epochs=15, 
                batch_size=128):
    
    test_predicts_list = []
    print("Starting to train models...")
    fold_size = len(data) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(data)

        print("Fold {0}".format(fold_id))
        
        train_x = np.concatenate([data[:fold_start], data[fold_end:]])
        train_xp = np.concatenate([data_post[:fold_start], data_post[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = data[fold_start:fold_end]
        val_xp = data_post[fold_start:fold_end]
        val_y = y[fold_start:fold_end]
        
        file_path="{}/capsule_fold{}.h5".format(output_dir, fold_id)
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        RocAuc = RocAucEvaluation(validation_data=([val_x, val_xp], val_y), interval=1)
        callbacks_list = [checkpoint, early, RocAuc] 

        hist = model.fit([train_x, train_xp], train_y, epochs=num_epochs, batch_size=batch_size, shuffle=True, 
                         validation_data=([val_x, val_xp], val_y), callbacks = callbacks_list, verbose=1)
        model.load_weights(file_path)
        best_score = min(hist.history['val_loss'])
        
        print("Fold {0} loss {1}".format(fold_id, best_score))
        print("Predicting validation...")
        val_predicts_path = "{}/capsule_val_predicts{}.npy".format(output_dir, fold_id)
        val_predicts = model.predict([val_x, val_xp], batch_size=1024, verbose=1)
        np.save(val_predicts_path, val_predicts)
        
        print("Predicting results...")
        test_predicts_path = "{}/capsule_test_predicts{}.npy".format(output_dir, fold_id)
        test_predicts = model.predict([test_data, test_data_post], batch_size=1024, verbose=1)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)
        
        test_predicts_am = np.zeros(test_predicts_list[0].shape)
        for fold_predict in test_predicts_list:
            test_predicts_am += fold_predict

        test_predicts_am = (test_predicts_am / len(test_predicts_list))
        return test_predicts_am
        
def main():
    parser = argparse.ArgumentParser(description='Capsule Network')
    parser.add_argument('-d', '--dataset', help='Dataset - agnews, toxic, imdb, yelp_polarity or yelp', required=True)
    parser.add_argument('-dir','--data_dir', help='Path to data directory', required=True)
    parser.add_argument('-e','--embedding_path', help='Path to pretrained GloVe embeddings', required=True)
    parser.add_argument('-o', '--output_dir', help='Path to output directory', required=True)
    
    parser.add_argument('-m', '--model', help='Model Type - base or large', default='base')
    parser.add_argument('-use_kfold', '--use_kfold', help='Use kfold for CV', default=True)
    parser.add_argument('-num_fold', '--num_fold', help='Number of folds for CV', default=10)
    parser.add_argument('-valid_ratio', '--valid_ratio', help='Validation set percentage', default=0.1)
    parser.add_argument('-num_epochs', '--num_epochs', help='Number of epochs', default=15)
    parser.add_argument('-batch_size', '--batch_size', help='Batch size', default=128)    
    
    parser.add_argument('-max_len', '--max_len', help='Maximun length of text', default=150)
    parser.add_argument('-max_features', '--max_features', help='Maximun number of words', default=100000)
    parser.add_argument('-spatial_dropout', '--spatial_dropout', help='Spatial dropout rate', default=0.4)
    parser.add_argument('-num_capsule', '--num_capsule', help='Number of capsules', default=10)
    parser.add_argument('-dim_capsule', '--dim_capsule', help='Dimension of capsule', default=16)
    parser.add_argument('-routings', '--routings', help='Routings', default=5)
    parser.add_argument('-gru_units', '--gru_units', help='Number of GRU units in the model', default=128)
    parser.add_argument('-max_pool', '--max_pool', help='Use global max pooling', default=False)
    parser.add_argument('-dropout', '--dropout', help='Dropout rate', default=0.25)
    parser.add_argument('-act', '--act', help='Activation at last layer - sigmoid or softmax', default='sigmoid')
    
    args = parser.parse_args()
    
    word_index, train_data_pre, train_data_post, y, test_data_pre, test_data_post = load_data(args.dataset, 
                                                                                              args.data_dir, 
                                                                                              args.max_len, 
                                                                                              args.max_features)
    
    embedding_matrix = load_embeddings(args.embedding_path, word_index, args.max_features)
    
    if args.model == 'base':
        model = CapsuleNetwork(embedding_matrix, 
                               max_len=args.max_len, 
                               max_features=args.max_features, 
                               embed_size=embedding_matrix.shape[1],
                               spatial_dropout_rate=args.spatial_dropout, 
                               gru_units=args.gru_units, 
                               num_capsule=args.num_capsule, 
                               dim_capsule=args.dim_capsule, 
                               routings=args.routings, 
                               dropout_rate=args.dropout,
                               max_pool=args.max_pool,
                               num_class=y.shape[1], 
                               act=args.act)
    else:
        model = CapsuleNetworkLarge(embedding_matrix, 
                                    max_len=args.max_len, 
                                    max_features=args.max_features, 
                                    embed_size=embedding_matrix.shape[1],
                                    spatial_dropout_rate=args.spatial_dropout, 
                                    gru_units=args.gru_units, 
                                    dropout_rate=args.dropout,
                                    num_class=y.shape[1], 
                                    act=args.act)

        
    if args.use_kfold:
        test_predicts = train_folds(model, 
                                    train_data_pre, 
                                    train_data_post, 
                                    y, 
                                    test_data_pre, 
                                    test_data_post, 
                                    args.output_dir, 
                                    args.num_fold, 
                                    args.num_epochs, 
                                    args.batch_size)
    else:
        test_predicts = train(model, 
                              train_data_pre, 
                              train_data_post, 
                              y, 
                              test_data_pre, 
                              test_data_post, 
                              args.output_dir, 
                              args.valid_ratio, 
                              args.num_epochs, 
                              args.batch_size)
    
    save_predictions(test_predicts, args.dataset, args.output_dir)

        
if __name__ == '__main__':
    main()
