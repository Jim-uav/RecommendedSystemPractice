# coding:utf-8
import random
import pandas as pd


def readfile():
    """
     # 读取文件中的用户、物品、行为数据
    """
    # header = ['user_id', 'item_id', 'rating', 'timestamp']
    # df = pd.read_csv("D:/SourceCode/Python/DataSet/ml-1m/ml-1m/ratings.dat", sep='::', names=header,engine='python')
    # df = pd.read_table("D:/SourceCode/Python/DataSet/ml-1m/ml-1m/ratings.dat", sep='::', header = None, names=header)
    # n_users = df.user_id.unique().shape[0]
    # n_items = df.item_id.unique().shape[0]
    # print 'all users is :' + str(n_users), 'all items is :' + str(n_items)
    # train, test = SplitData(df, 8, 2, 0.5)
    # return train, test

    data=[]
    # for line in open("D:/SourceCode/Python/DataSet/ml-1m/ml-1m/ratings.dat"):
        # user_id, item_id, rating, _ = line.split('::')
    for line in open("D:/SourceCode/Python/DataSet/ml-100k/ml-100k/u.data"):
        user_id, item_id, rating, _ = line.split()
        data.append((user_id, item_id, int(rating)))    # rating原本是字符串数字，应该转为整数型int
    # print data
    train, test = SplitData(data, 8, 4, 100)
    return train, test


def SplitData(data, M, k, seed):
    """
    # Page-42
    # 把数据集分成训练集和测试集
    :input: M是分的组数，seed是随机数种子，k是实验系数，在区间[0,M-1]，每次实验都取不一样的值
    :output:训练集和测试集
    """
    train = {}
    test = {}
    random.seed(seed)
    for user, item, rating in data:
        if random.randint(0, M) == k:
            test.setdefault(user, {})     # setdefault(user,{})函数，设置键为user时的默认值为字典null。否则，如果键为user的值为空时，会报错。
            test[user][item] = rating
        else:
            train.setdefault(user, {})
            train[user][item] = rating
    return train, test
