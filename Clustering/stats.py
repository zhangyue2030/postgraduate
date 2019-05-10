import collections
from scipy.spatial import distance
from operator import add


# 用于计算簇内分散的个体
def calc_ind_within_cluster_scatter(data, centers, count, k):
    data_length = len(data)
    within_cluster_scatter = []
    total = 0
    for i in range(0, k):
        for j in range(0, data_length):
            if data[j][5] == i:
                total += distance.euclidean(data[j][0:4:1], centers[i])
        within_cluster_scatter.append((float(1) / count[i]) * total)
    return within_cluster_scatter


# 用于计算簇内总散射
def calc_ov_within_cluster_scatter(data, centers, k):
    data_length = len(data)
    wcs = 0
    temp_tot = 0
    for i in range(0, k):
        temp_tot = 0
        for j in range(0, data_length):
            if data[j][5] == i:
                temp_tot += ((distance.euclidean(data[j][0:4:1], centers[i])) ** float(2))
    wcs += temp_tot
    return wcs


# 用于计算簇之间总体差异的函数
def calc_between_cluster_variance(data, centers, count, k):
    data_length = len(data)
    bcv = 0
    mean = [0.0, 0.0, 0.0, 0.0]
    for i in range(0, data_length):
        mean = list(map(add, data[i][0:4:1], mean))
    for i in range(0, 4):
        mean[i] /= data_length
    for i in range(0, k):
        bcv += (count[i] * ((distance.euclidean(centers[i], mean)) ** float(2)))
    return bcv


# 计算簇间距（欧几里德距离）
def calc_between_cluster_spread(center1, center2):
    return distance.euclidean(center1, center2)


# 计算度量聚的好的簇
def calc_cluster_measure(scatter1, scatter2, spread):
    return (scatter1 + scatter2) / spread


# 计算Davies Bouldin索引.
def calc_dbi(k, d):
    return (float(1) / float(k)) * (float(3) * float(d))


# 函数来计算Davies Bouldin索引.
def calc_chi(N, k, ssb, ssw):
    return (ssb / ssw) * ((N - k) / (k - 1))


# 使用简单指标评估簇的功能. --2019/4/17 zhangyue
def evaluate_clusters_simple(data, count, k):
    data_length = len(data)
    evaluation = []
    for i in range(0, k):
        counter = collections.Counter()
        for j in range(0, data_length):
            if data[j][5] == i:
                counter[data[j][4]] += 1
        evaluation.append(counter)
    # print("类别 1: ", dict(evaluation[0]))
    # print("类别 2: ", dict(evaluation[1]))
    # print("类别 3: ", dict(evaluation[2]))
    for i in range(0, k):
        dominant_label = max(evaluation[i], key=evaluation[i].get)
        print("簇", i + 1, "的主标签为'", dominant_label, "'", int(count[i]), "数据点中有",
              evaluation[i][dominant_label], "个. (", round(
            float((float(evaluation[i][dominant_label]) / count[i]) * float(100))), "% )")

# 对一组簇执行Davies-Bouldin Index,DBI评估.
def evaluate_clusters_complex(data, centers, count, k):
    within_cluster_scatter = calc_ind_within_cluster_scatter(data, centers, count, k)
    sep1_2 = calc_between_cluster_spread(centers[0], centers[1])
    sep1_3 = calc_between_cluster_spread(centers[0], centers[2])
    sep2_3 = calc_between_cluster_spread(centers[1], centers[2])
    cm1_2 = calc_cluster_measure(within_cluster_scatter[0], within_cluster_scatter[1], sep1_2)
    cm1_3 = calc_cluster_measure(within_cluster_scatter[0], within_cluster_scatter[2], sep1_3)
    cm2_3 = calc_cluster_measure(within_cluster_scatter[1], within_cluster_scatter[2], sep2_3)
    dbi = calc_dbi(k, max(cm1_2, cm1_3, cm2_3))
    print("簇集合的Davies Bouldin Index（越低越好）：", round(dbi, 4))
    ssw = calc_ov_within_cluster_scatter(data, centers, k)
    ssb = calc_between_cluster_variance(data, centers, count, k)
    chi = calc_chi(len(data), k, ssb, ssw)
    print("簇集合的Calinski Harabasz Index（越高越好）：", round(chi, 4))