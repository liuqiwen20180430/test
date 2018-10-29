import time
import pandas as pd
import numpy as np
import jieba as jb
import jieba.analyse
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
goods_data = pd.read_csv('goods_data2.csv', usecols=['title', 'category3'])
jb.load_userdict("userdic.dic")
jb.analyse.set_stop_words("stop.dic")
def load_data(data):
    data1 = jb.cut(data)
    data11 = ''
    for item in data1:
        data11+=item+' '
    return data11

def countVect_transformer():
    goods_data2 = goods_data.T
    content = goods_data2.values[0]
    train_text = []
    for i in range(0,len(content)):
        this_data = load_data(str(content[i]))
        train_text.append(this_data)
    count_vect = CountVectorizer()
    return count_vect.fit(train_text)

def tfidfVect_transformer(count_vect_trans):
    goods_data2 = goods_data.T
    content = goods_data2.values[0]
    train_text = []
    for i in range(0, len(content)):
        this_data = load_data(str(content[i]))
        train_text.append(this_data)
    count_vect=count_vect_trans.transform(train_text)
    return TfidfTransformer().fit(count_vect)

def study(tfidf_vect,train_x_counts,train_class):
    #tfidf模型
    train_x_data = tfidf_vect.transform(train_x_counts)
    #创建模型,训练数据
    model = MultinomialNB().fit(train_x_data,train_class)
    return  model


def prepare_data(count_vect,goods):
    # 读取数据
    goods_data2 = goods.T
    content = goods_data2.values[0]
    # print(content[:10])
    class_type = goods_data2.values[1]
    # print(class_type[:10])
    train_text = []
    train_class = []
    for i in range(0,len(content)):
        this_data = load_data(str(content[i]))
        train_text.append(this_data)
        train_class.append(class_type[i])
    train_x_counts =count_vect.transform(train_text)
    return train_x_counts,train_class

#结果测试的方法
def run():
    #====================以下为测试识别的准确率的代码
    #准备数据
    begin = time.time()
    train,test= train_test_split(goods_data ,test_size=0.3)
    count_vect = countVect_transformer()
    tfidf_vect = tfidfVect_transformer(count_vect)
    x_train,y_train = prepare_data(count_vect,train)
    #训练模型
    if os.path.exists("model.pkl"):
        model=joblib.load("model.pkl")
    else:
        model = study(tfidf_vect,x_train,y_train)
        joblib.dump(model, 'model.pkl')

    study_end =time.time()
    print(f'模型训练完成,总共耗时{study_end-begin}秒,接下来进行结果预测')
    # test_data = pd.read_csv(open("需要修正类目的商品.csv",encoding="utf-8"), usecols=['title','category3'])
    # print("read csv 成功")
    x_test,y_test=prepare_data(count_vect,test)
    #对学习效果进行预测
    pridect_data = model.predict(x_test)

    count = 0
    right = 0
    for idx,pt,yt in zip(test.index,pridect_data,y_test):
        name= test.loc[idx]['title']
        print(f'商品标题为: {name} 预测值为: {pt},真实值为: {yt}')
        count+=1
        if pt==yt:
            print('预测正确!!!')
            right+=1
        else:
            print('预测错误!!!')
        print('*****')

    print(f'准确率为: {right/count}')

    end = time.time()
    print('*********' * 5)
    print(f'本次总共耗时{end-begin}秒')



if __name__ == '__main__':
    run()










    #
    # for i in range(1,11):
    #     print('==========='*5)
    #     print(f'本次的测试数据量为{0.05*i}')
    #     x_train,x_test,y_train,y_test = train_test_split(traindata, labels,test_size=0.05*i,random_state=20)
    #     count=0
    #     right=0
    #     for test,rst in zip(x_test,y_test):
    #         bayes_rst= bayes(test,x_train,y_train)
    #         count+=1
    #         if bayes_rst == rst:
    #             right+=1
    #     print(f'准确率为:{right/count}')
    # end = time.time()
    # print('*********' * 5)
    # print(f'本次总共耗时{end-begin}秒')




