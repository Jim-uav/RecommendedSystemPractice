# coding: utf-8
import math

"""
 # 用户或者物品的相似度计算方法：利用用户的行为数据来计算相似度，而不是利用物品的描述属性计算
"""

"""
# 余弦相似度（cosine similarity）
"""
"""
 # 用户相似度计算
"""


def userSimilarity(train):
    """
    # 1、计算用户相似度方法1
    # P46,该方法复杂度是O(n*n)，当用户量很大时，非常耗时。
    :param train:
    :return W:
    """
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            W.setdefault(u, {})
            W[u][v] = len(set(train[u].keys()) & set(train[v].keys()))
            W[u][v] /= math.sqrt(len(train[u])*len(train[v])*1.0)
    return W


def userSimilarity2(train):
    """
    # P46
    # 2、计算用户相似度方法2，改进法
    # 建立物品到用户的倒排表，排除用户都没有产生过行为的物品，即|N(u)& N(v)|=0。
    # 分为3个步骤：1、把用户到物品的索引转为物品到用户的索引；2、计算用户u和v同时属于同一个物品的个数C[u][v]；3、计算用户相似度
    # 1）、把用户到物品的索引转为物品到用户的索引
    :param train:
    :return:
    """
    item_users = dict()
    for user, items in train.items():    # 从下面的循环可以看出，train和item_users数据类型很像，但其值应该也是字典。
        for item in items.keys():  # 此时的items应该是字典。
            if item not in item_users:
                item_users[item] = set()    # item_users是字典，键是物品名，但里面的值是集合，其包含对该物品有过行为的所有用户。
            item_users[item].add(user)
    # 2）、计算用户u和v同时属于同一个物品的个数C[u][v]
    C = dict()
    N = dict()
    for item, users in item_users.items():
        for u in users:   # 此时的users应该是集合，因此不需要users.keys()来索引每个user
            N.setdefault(u, 0)   # 为字典N设置键为u时的默认值为0，不设置的话，当键为u时，其值可能为空，下一步就会报错。
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C.setdefault(u, {})        # 为字典C设置键为u时的默认值为空字典，不设置的话，当键为u时，其值可能为空，下一步就会报错。
                C[u].setdefault(v, 0)      # 为字典C[u]设置键为v时的默认值为0，不设置的话，当其键为v时，其值可能为空，下一步就会报错。
                C[u][v] += 1
    # 3）、计算用户相似度
    W=dict()
    for u, vs in C.items():
        W.setdefault(u, dict())        # 也需要设置默认值
        for v, uvalue in vs.items():
            W[u][v] = uvalue/math.sqrt(N[u]*N[v]*1.0)
    return W


def userSimilarity3(train):
    """
    # 3、计算用户相似度方法3，改进法：User-IIF算法
    # P49, 增加了对数表达式，用于惩罚用户u和v共同兴趣列表中热门物品的他们相似度的影响。
    :param train:
    :return:
    """
    item_users=dict()
    for u, items in train.items():
        for item in items.keys():
            if item not in item_users:
                item_users[u]=set()
            item_users[u].add(u)
    C=dict()
    N=dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1 / math.log(1 + len(users))
    W=dict()
    for u, vs in C.items():
        for v, value in vs.items():
            W[u][v] += value/math.sqrt(N[u] * N[v] * 1.0)
    return W

"""
 # P53
 # 物品相似度计算
 # input:train依然是用户、物品、行为的数据存储格式
 # 也需要计算出用户-物品的倒排表。
"""


def itemSimilarity(train):
    """
    # P53
    # 1、计算物品相似度方法1，用户到物品的倒排表
    # 建立用户到物品的倒排表，排除用户都没有产生过行为的物品，即|N(i)& N(j)|=0。
    :param train:
    :return:
    """
    C=dict()
    N=dict()
    for user, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
    W=dict()
    for i, js in C.items():
        for j, cij in js.items():
                W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W


def itemSimilarity2(train):
    """
    # P58
    # 2、计算物品相似度2：增加对活跃用户的惩罚
    # ItemCF-IUF算法
    :param train:
    :return:
    """
    N = dict()
    C = dict()
    for user, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1/math.log(1 + len(items) * 1.0)
    W = dict()
    for i,js in C.items():
        for j, cij in js.items():
            W[i][j] = cij/math.sqrt(N[i] * N[j])
    return W


def itemSimilarity3(train):
    """
    # P58
    # 3、计算物品相似度矩阵3：增加对活跃用户的惩罚，且对物品相似度矩阵按最大值归一化。
    # ItemCF-Norm算法
    :param train:
    :return:
    """
    N = dict()
    C = dict()
    for user, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1/math.log(1 + len(items) * 1.0)
    W = dict()
    for i,js in C.items():
        for j, cij in js.items():
            W[i][j] = cij/math.sqrt(N[i] * N[j])
    for i,js in W.items():
        for j, jv in js.items():
            W[i][j] /= max(js)    # ????
    return W


def itemSimilarity4(train, alpha):
    """
    # P63
    # 4、计算物品相似度矩阵4：在分母加大对热门物品的惩罚，通过这种方法可以适当的牺牲准确率和召回率的情况下，显著提升结果的覆盖率和新颖性。
    # ItemCF-Norm算法
    :param train:
    :param alpha:
    :return:
    """
    C=dict()
    N=dict()
    for user, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
    W = dict()
    for i, js in C.items():
        for j, cij in js.items():
                W[i][j] = cij / math.sqrt(N[i]**(1-alpha) * N[j]**alpha)   # 在分母加大了惩罚。
    return W

"""
 # 皮尔逊相关系数（Pearson correlation）
"""


"""
 # 欧式距离法

"""

