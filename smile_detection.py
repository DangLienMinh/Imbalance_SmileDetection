import cv2
import numpy as np
import dlib
from imutils import face_utils
import tensorflow as tf
import os
import argparse
import distutils
from imutils.face_utils import FaceAligner
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import KFold, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, average_precision_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import interp
import itertools
from keras import backend as K

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier


nb_classes = 2
img_rows = 64
img_cols = 64
channel = 1

path = "genki4k"
data_path = "files"
face_path = "face_type1"
label_path = "labels.txt"


def smile_cnn():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(9, 9), input_shape=(
        img_rows, img_cols, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=8, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))

    learning_rate = 0.0001
    momentum = 0.9
    decay = 0.0005
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
  
    return model


def smile_v2():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=(
        img_rows, img_cols, 1), activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print model.summary()

    return model


def read_data(data_fn, label_fn):
    y = []
    x = []
    idx = []
    for filename in os.listdir(data_fn):
        idx.append(int(filename[4:8]))
        data = cv2.imread(data_fn+"/"+filename, 0)
        x.append(data)

    x = np.reshape(x, (-1, img_rows, img_cols, channel))
    x = x.astype(np.float32)/255.

    with open(label_fn) as f:
        content = f.readlines()
    content = [i.strip() for i in content]

    for id in idx:
        data = content[id-1]
        y.append(int(data.split(' ')[0]))

    y_one_hot = np_utils.to_categorical(y, nb_classes).astype(np.float32)
    y_one_hot = y_one_hot.astype(np.float32)
    y = np.asarray(y)
    return x, y_one_hot, y


def face_detecton(data_path, label_fn):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredLeftEye=(0.25, 0.25),
                     desiredFaceWidth=64, desiredFaceHeight=64)
    count = 0

    for filename in os.listdir(data_path):
        file = data_path+"/"+filename
        idx = int(filename[4:8])

        image = cv2.imread(file)
        image = cv2.resize(image, (512, 512))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            (x, y, width, height) = face_utils.rect_to_bb(rect)
            face_rect = image[y:y + width, x:x + height]

            try:

                face_rect = cv2.resize(face_rect, (64, 64))
                faceAligned = fa.align(image, gray, rect)
                cv2.rectangle(image, (x, y), (x + width,
                                              y + height), (255, 0, 0), 2)
               
                cv2.imshow("Face", face_rect)
                cv2.imshow("Face Aligned", faceAligned)

            except cv2.error:
                count = count + 1
                print "Error at {0}".format(filename)

                pass

        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print "Number error file:{0}".format(count)


class_name = ["No Smile", "Smile"]


def train_model(X, Y, class_weight, title_model="Smile CNN"):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    cvscore = []
    auc_score = []
    count = 0
    for train, test in kfold.split(X, Y):
        y_test = Y[test]
        y_train = Y[train]
        y_test = np_utils.to_categorical(y_test, nb_classes).astype(np.float32)
        y_train = np_utils.to_categorical(
            y_train, nb_classes).astype(np.float32)

        model = smile_cnn()
        count_none_smile = (y_test == [1, 0]).sum() / 2
        count_smile = (y_test == [0, 1]).sum() / 2
     
        print "count none smile: %s" % count_none_smile
        print "count smile: %s" % count_smile

        model.fit(X[train], y_train, nb_epoch=100, batch_size=50,
                  verbose=0, validation_split=0.1, shuffle=True)

        y_prob = model.predict(X[test])
        fpr, tpr, thresholds = roc_curve(Y[test], y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        scores = model.evaluate(X[test], y_test, verbose=0)

        model_name = "model_origin/" + title_model + \
            "_acc_{0}".format(scores[1]) + "_{0}.h5".format(count)
        # model.save_weights(model_name)

        print("%s: %.2f%%" % ("roc auc", roc_auc * 100))
        print("%s: %.2f%%" % ("acc", scores[1] * 100))
        auc_score.append(roc_auc * 100)
        cvscore.append(scores[1]*100)

        count = count + 1
    

    print("AUC Score:%.2f%% (+/- %.2f%%)" %
          (np.mean(auc_score), np.std(auc_score)))
    print("ACC:%.2f%% (+/- %.2f%%)" % (np.mean(cvscore), np.std(cvscore)))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def predict_model(X, Y, result_file, title_text):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    cvscore_base = []
    cvscore_svm = []
    cvscore_xgb = []
    cvscore_ada = []
    auc_base = []
    auc_svm = []
    auc_ada = []
    auc_xgb = []

    tprs_base = []
    tprs_svm = []
    tprs_ada = []
    tprs_xgb = []
    mean_fpr = np.linspace(0, 1, 100)

    count = 0
    for train, test in kfold.split(X, Y):
        y_train = Y[train]
        y_test = Y[test]
        base_model = smile_cnn()
        count_none_smile = (y_test == 0).sum()
        count_smile = (y_test == 1).sum()

        base_model.load_weights(
            "model_sgd/model_ratio_1.0_roc_auc_0.667703366763_0.h5")
        # Layer extraction for model v1
        intermediate_layer_model = Model(
            inputs=base_model.input, outputs=base_model.layers[7].output)
        # Layer extraction for model v2
        #intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.layers[11].output)
    
        train_feature = intermediate_layer_model.predict(X[train])
        test_feature = intermediate_layer_model.predict(X[test])

        # base model
        proba_ = base_model.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y_test, proba_[:, 1])
        tprs_base.append(interp(mean_fpr, fpr, tpr))
        tprs_base[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_base.append(roc_auc)
        # plt.figure(4)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc))

        # Test SVM
        model_svm = svm.SVC(kernel='linear', C=1, probability=True)
        model_svm.fit(train_feature, y_train)
        # Cal AUC and draw plot SVM
        proba_ = model_svm.predict_proba(test_feature)
        cnf_matrix = confusion_matrix(Y[test], model_svm.predict(test_feature))
        with open(result_file, 'a') as f:
            print "SVM"
            print cnf_matrix
            f.write('SVM\n')
            f.write('{0}\n'.format(cnf_matrix))
        fpr, tpr, thresholds = roc_curve(y_test, proba_[:, 1])
        tprs_svm.append(interp(mean_fpr, fpr, tpr))
        tprs_svm[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_svm.append(roc_auc)
        # plt.figure(1)
        # plt.plot(fpr,tpr,lw=1, alpha= 0.3,  label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc))

        # Test ADA
        model_ada = AdaBoostClassifier(DecisionTreeClassifier(
            max_depth=2), algorithm='SAMME', n_estimators=200)
        model_ada.fit(train_feature, Y[train])
        # Cal AUC and draw plot ADA
        proba_ = model_ada.predict_proba(test_feature)
        cnf_matrix = confusion_matrix(Y[test], model_ada.predict(test_feature))
        with open(result_file, 'a') as f:
            print "ADA"
            print cnf_matrix
            f.write('ADA\n')
            f.write('{0}\n'.format(cnf_matrix))
        fpr, tpr, thresholds = roc_curve(Y[test], proba_[:, 1])
        tprs_ada.append(interp(mean_fpr, fpr, tpr))
        tprs_ada[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_ada.append(roc_auc)
        # plt.figure(2)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc) )

        # Test XGB
        model_xgb = XGBClassifier()
        model_xgb.fit(train_feature, Y[train])
        # Cal AUC and draw plot XGB
        proba_ = model_xgb.predict_proba(test_feature)
        cnf_matrix = confusion_matrix(Y[test], model_xgb.predict(test_feature))
        with open(result_file, 'a') as f:
            print "XGB"
            print cnf_matrix
            f.write('XGB\n')
            f.write('{0}\n'.format(cnf_matrix))
        fpr, tpr, thresholds = roc_curve(Y[test], proba_[:, 1])
        tprs_xgb.append(interp(mean_fpr, fpr, tpr))
        tprs_xgb[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_xgb.append(roc_auc)
        # plt.figure(3)
        # plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count, roc_auc))

        # Result
        with open(result_file, 'a') as f:
            print "count none smile: %s" % count_none_smile
            print "count smile: %s" % count_smile
            print "BASE:{0}".format(auc_base[-1])
            print "SVM:{0}".format(auc_svm[-1])
            print "ADA:{0}".format(auc_ada[-1])
            print "XGB:{0}".format(auc_xgb[-1])

            f.write("Fold:{0}\n".format(count))
            f.write("count none smile: {0}\n".format(count_none_smile))
            f.write("count smile: {0}\n".format(count_smile))
            f.write("AUC Base:{0}\n".format(auc_base[-1]))
            f.write("AUC SVM:{0}\n".format(auc_svm[-1]))
            f.write("AUC ADA:{0}\n".format(auc_ada[-1]))
            f.write("AUC XGB:{0}\n".format(auc_xgb[-1]))

        count = count + 1

    # Base plot
    # plt.figure(4)
    mean_tpr_base = np.mean(tprs_base, axis=0)
    mean_tpr_base[-1] = 1.0
    mean_auc_base = auc(mean_fpr, mean_tpr_base)
    std_auc_base = np.std(auc_base)
    # plt.plot(mean_fpr, mean_tpr_base, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_base, std_auc_base),
    #          lw=2, alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve SMILE-CNN with Softmax classifier')
    # plt.legend(loc="lower right")
    # plt.savefig('ROC %s with %s.png' % (title_text, 'Base'))
    # plt.clf()

    # #SVM plot
    # plt.figure(1)
    mean_tpr_svm = np.mean(tprs_svm, axis=0)
    mean_tpr_svm[-1] = 1.0
    mean_auc_svm = auc(mean_fpr, mean_tpr_svm)
    std_auc_svm = np.std(auc_svm)
    # plt.plot(mean_fpr,mean_tpr_svm,color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_svm, std_auc_svm),
    #          lw=2,alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve SMILE-CNN with SVM classifier')
    # plt.legend(loc="lower right")
    # plt.savefig('ROC %s with %s.png'%(title_text,'SVM'))
    # plt.clf()
    #
    # #ADA plot
    # plt.figure(2)
    mean_tpr_ada = np.mean(tprs_ada, axis=0)
    mean_tpr_ada[-1] = 1.0
    mean_auc_ada = auc(mean_fpr, mean_tpr_ada)
    std_auc_ada = np.std(auc_ada)
    # plt.plot(mean_fpr, mean_tpr_ada, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_ada, std_auc_ada),
    #          lw=2, alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve SMILE-CNN with Ada classifier')
    # plt.legend(loc="lower right")
    # plt.savefig('ROC %s with %s.png' % (title_text, 'ADA'))
    # plt.clf()
    #
    # # XGB plot
   # plt.figure(3)
    mean_tpr_xgb = np.mean(tprs_xgb, axis=0)
    mean_tpr_xgb[-1] = 1.0
    mean_auc_xgb = auc(mean_fpr, mean_tpr_xgb)
    std_auc_xgb = np.std(auc_xgb)
    # plt.plot(mean_fpr, mean_tpr_xgb, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_xgb, std_auc_xgb),
    #          lw=2, alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # #plt.title('ROC %s \n with %s' % (title_text, 'XGB'))
    # plt.legend(loc="lower right")
    # plt.savefig('ROC %s with %s.png' % (title_text, 'XGB'))
    # plt.clf()
    #
    #
    with open(result_file, 'a') as f:
        print("Mean AUC Base:%.2f%% (+/- %.2f%%)" %
              (mean_auc_base, std_auc_base))
        print("Mean AUC SVM:%.2f%% (+/- %.2f%%)" % (mean_auc_svm, std_auc_svm))
        print("Mean AUC ADA:%.2f%% (+/- %.2f%%)" % (mean_auc_ada, std_auc_ada))
        print("Mean AUC XGB:%.2f%% (+/- %.2f%%)" % (mean_auc_xgb, std_auc_xgb))

        f.write("Mean AUC Base:%.2f%% (+/- %.2f%%)\n" %
                (mean_auc_base, std_auc_base))
        f.write("Mean AUC SVM:%.2f%% (+/- %.2f%%)\n" %
                (mean_auc_svm, std_auc_svm))
        f.write("Mean AUC ADA:%.2f%% (+/- %.2f%%)\n" %
                (mean_auc_ada, std_auc_ada))
        f.write("Mean AUC XGB:%.2f%% (+/- %.2f%%)\n\n\n" %
                (mean_auc_xgb, std_auc_xgb))


def generate_random_data(data, label, n_part=.1):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    label = label[indices]
    return data[:int((len(data)+1)*n_part)], label[:int((len(label)+1)*n_part)]
 


def train():

    X = np.load("data_1.npy")
    Y = np.load("label_1.npy")
    class_totals = Y.sum(axis=0)
    class_weight = class_totals.max() / class_totals

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]
    train_model(X, Y, class_weight)


def train2():
    X = np.load('data_1.npy')
    Y = np.load('label_1.npy')

    indice_smile = np.where(Y == 1)
    indice_no_smile = np.where(Y == 0)
    x_smile = X[indice_smile]
    x_no_smile = X[indice_no_smile]
    y_smile = Y[indice_smile]
    y_no_smile = Y[indice_no_smile]

    for i in np.arange(0.1, 0.2, 0.1):
        x_sample_smile, y_sample_smile = generate_random_data(
            x_smile, y_smile, n_part=i)
        X_imbalance = np.concatenate([x_no_smile, x_sample_smile])
        Y_imbalance = np.concatenate([y_no_smile, y_sample_smile])

        indices = np.arange(len(X_imbalance))
        np.random.shuffle(indices)
        X_imbalance = X_imbalance[indices]
        Y_imbalance = Y_imbalance[indices]
        class_totals = Y_imbalance.sum(axis=0)
        class_weight = class_totals.max() / class_totals
        title_model = 'model_ratio_{0}'.format(i)
        train_model(X_imbalance, Y_imbalance, class_weight, title_model)


def test():
    X = np.load("data_1.npy")
    Y = np.load("label_1.npy")

    indice_smile = np.where(Y == 1)
    indice_non_smile = np.where(Y == 0)
    x_smile = X[indice_smile]
    y_smile = Y[indice_smile]
    x_non_smile = X[indice_non_smile]
    y_non_smile = Y[indice_non_smile]

    result_file = "result_report_imbalanced_origin_11.txt"

    # get sample smile dataset (1/10, 2/10,...,10/10)
    for i in np.arange(1.0, 1.1, 0.1):
        x_sample_smile, y_sample_smile = generate_random_data(
            x_smile, y_smile, n_part=i)
        X_imbalance = np.concatenate([x_non_smile, x_sample_smile])
        Y_imbalance = np.concatenate([y_non_smile, y_sample_smile])
        with open(result_file, 'a') as f:
            print "Shape of imbalance Data:", X_imbalance.shape
            print "Shape of imbalance label:", Y_imbalance.shape
            f.write("Shape of imbalance data:{0}\n".format(X_imbalance.shape))
            f.write("Shape of imbalanced label:{0}\n".format(
                Y_imbalance.shape))
            f.write("Len of smile data:{0}\n".format(x_sample_smile.shape))
            f.write("Len of non smile data:{0}\n".format(x_non_smile.shape))

        # shuffle data
        indices = np.arange(len(X_imbalance))
        np.random.shuffle(indices)
        X_imbalance = X_imbalance[indices]
        Y_imbalance = Y_imbalance[indices]
        title_text = "Num dataset {0} \n num smile {1} - ratio {3}, num non smile {2}".format(X_imbalance.shape[0],
                                                                                              x_sample_smile.shape[0],
                                                                                              x_non_smile.shape[0], i)
        predict_model(X_imbalance, Y_imbalance, result_file, title_text)


if __name__ == "__main__":
    seed = 10
    np.random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('--test', action='store_true', help='test flag')
    parser.add_argument('--generate_data',
                        action='store_true', help='generate face data')

    args = parser.parse_args()
    if args.train:
        print "train"
        train()
    elif args.test:
        print "testing"
        test()
    elif args.generate_data:
        print "generate data"
        data_fn = path + "/" + data_path
        label_fn = path + "/" + label_path
        face_detecton(path + "/" + data_path, path + "/" + label_path)
        #X, Y_one_hot, Y = read_data(face_path, path + "/" + label_path)
        # np.save("data_1.npy", X)
        # np.save("label_encoding_1.npy", Y_one_hot)
        # np.save("label_1.npy", Y)
