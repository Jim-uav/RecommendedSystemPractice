# coding: utf-8
import dataset_process as dsp
import similarity_function as sf
import recommended_system as rs
import evaluation_indicators as ei

train, test = dsp.readfile()
# user_similar = sf.userSimilarity(train)
user_similar = sf.userSimilarity2(train)
user = '345'
rank = rs.recommendation1(user, train, user_similar, 3)
# print rank
# 打印：对比分析test集中user的行为数据和train集训练得到的用户user可能的行为数据rank。
# 对比：train训练得到的用户user的推荐列表rank & 用户user在test集中的对物品i的行为动作，即评分多少。
# for i, rvi in rank.items():
#     items = test.get(user, {})
#     rating = items.get(i, 0)
#     print "%5s: %.4f -- %.4f" % (i, rvi, rating)

# 计算出推荐算法的召回率、准确率、覆盖率和流行度
print "%3s%20s%20s%20s%20s" % ('K', 'recall', 'precision', 'coverage', 'popularity')
for k in [5, 10, 20]:
    recall = ei.recall(train, test, user_similar, k)
    precision = ei.precision(train, test, user_similar, k)
    coverage = ei.coverage(train, test, user_similar, k)
    popularity = ei.popularity(train, test, user_similar, k)
    print "%3d%19.3f%%%19.3f%%%19.3f%%%20.3f" % (k, recall * 100, precision * 100, coverage * 100, popularity)