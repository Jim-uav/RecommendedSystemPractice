# coding: utf-8
import recommended_system as rs
import math


def recall(train, test, W, N):
    """
     P43
    :1、召回率
    :train、test、N分别是训练集、测试集和推荐列表长度
    :描述：一个用户u可能对多个物品有过行为，从而数据集中用户u有多条行为记录，在将数据集进行分组时，
    训练集和测试集里都可能存在用户u的多条行为记录。利用训练集训练推荐系统，给出用户u的推荐结果，再和测试集中用户u的行为数据进行对比，
    检验算法推荐精度。
    """
    hit = 0
    all = 0
    for user in train.keys():
        # tu = test[user]
        tu = test.get(user, {})
        rank = rs.recommendation1(user, train, W, N)  #是推荐系统函数，返回用户u的N个推荐结果
        for item in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit/(all*1.0)


def precision(train, test, W, N):
    """
    P43
    # 2、准确率
    # train、test、N分别是训练集、测试集和推荐列表长度
    """
    hit = 0
    all = 0
    for user in train.keys():
        # tu=test[user]
        tu = test.get(user, {})
        rank = rs.recommendation1(user, train, W, N)
        for item in rank:
            if item in tu:
                hit += 1
        all += len(rank)
    return hit/(all*1.0)


def coverage(train, test, W, N):
    """
    P43
    # 3、覆盖率
    """
    recommend_items=set()
    all_items=set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank=rs.recommendation1(user, train, W, N)
        for item in rank:
            recommend_items.add(item)
    return len(recommend_items)/(len(all_items)*1.0)


def popularity(train, test, W, N):
    """
    P44
    4、流行度
    """
    item_popularity = dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item]=0
            item_popularity[item]+=1
    ret=0
    n=0
    for user in train.keys():
        rank = rs.recommendation1(user, train, W, N)
        for item in rank:
            ret += math.log(1+item_popularity[item])
            n += 1
    return ret/(n*1.0)

