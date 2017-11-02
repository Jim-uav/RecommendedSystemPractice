# coding: utf-8
import operator
"""
 # 计算用户u对物品i的感兴趣程度p(u,i)
"""


def getRecommenderSystem(user,N):
    return 0


def recommendation1(user, train, W, K):
    """
     P47
     描述：计算用户u对物品i的兴趣程度：基于用户相似度矩阵
     input:用户user，训练数据集train，用户相似度矩阵W，与用户相似度最接近的用户个数K
     output:用户u对各还没有产生过行为的物品i的感兴趣程度：字典变量rank
    """
    rank = dict()
    # both_items = train[user]   # 已经有过行为的物品，不需要计算用户对其的感兴趣程度
    both_items = train.get(user, {})
    # '从字典train中提取键为user的值，其值也是一个字典，为避免值为空，通过get(user, {})函数，当user键对应的值为空时，为其设置一个默认的空字典。'
    # for v, w in sorted(W[user].items(), key=operator.itemgetter(1), reverse=True)[:K]:
    for v, w in sorted(W[user].items(), key=lambda x: x[1], reverse=True)[:K]:
        for item, vr in train[v].items():
            if item in both_items:
                continue
            # rank[item] += w * vr
            rank.setdefault(item, 0)  # 设置默认值
            rank[item] += w
    # return rank
    return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:40])   # 返回用户user对物品感兴趣程度排名前40的物品，及其感兴趣度。


def recommendation2(user, train, W, K):
    """
     P55
     描述：计算用户u对物品i的兴趣程度：基于物品相似度矩阵
     input:用户user，训练数据集train，物品相似度矩阵W，与物品i相似度最接近的物品个数K
     output:基于用户u已经产生过行为的物品j的相似度，计算出用户u对各还没有产生过行为的物品i的感兴趣程度：字典变量rank
    """
    rank = dict()
    items = train(user)   # 已经有过行为的物品，不需要计算用户对其的感兴趣程度
    for j, rui in items.items():
        for i, wij in sorted(W[j].items(), operator.itemgetter(1), reverse=True)[:K]:
            if i in items:
                continue
            rank[i] += wij * rui
    return rank


def recommendation3(user, train, W, K):
    """
     P56
     描述：计算用户u对物品i的兴趣程度：基于物品相似度矩阵，并提供推荐解释，列出了每个历史物品对没有产生用户行为的物品的影响大小reason[i]
     input:用户user，训练数据集train，物品相似度矩阵W，与物品i相似度最接近的物品个数K
     output:基于用户u已经产生过行为的物品j的相似度，计算出用户u对各还没有产生过行为的物品i的感兴趣程度：字典变量rank
    """
    rank = dict()
    items = train[user]  # 用户已经产生过行为的物品，也就是历史行为
    for j, rui in items.items():
        for i, wij in sorted(W[user].items(), operator.itemgetter(1), reverse=True)[:K]:
            if i in items:
                continue
            rank[i].weight += wij * rui   # 基于用户的历史行为，计算用户对还没有产生行为的物品的兴趣程度。
            rank[i].reason[j] = wij * rui
    return rank
