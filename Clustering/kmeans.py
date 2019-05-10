import numpy
from scipy.spatial import distance
from operator import add, truediv


# 从数据中生成特征向量，设置参数D 数字列数 可以兼容这类不同列的数据集
def get_feat_vects(fname, D):
    feat_vects = []
    # 打开数据文件
    with open(fname) as feat_file:
        for line in feat_file:
            x = []
            count = 1
            # 用逗号和行拆分特征 注意：在官方下载的数据集最后两行会有空白行，考虑到数据中出现的空行，加上对x的判断
            for w in line.strip().split(','):
                if count < D:
                    # 将字符串表示转换为浮点数
                    x.append(float(w))
                else:
                    # 追加字符串标签
                    x.append(w)
                count += 1
                feat_vects.append(x)
    return feat_vects

# 使用欧几里得距离聚类数据
def cluster_data(centers, data, c):
    data_length = len(data)
    centers_length = len(centers)
    # 迭代每个特征向量
    for i in range(0, data_length):
        # 迭代每个聚类中心
        for j in range(0, centers_length):
            # 从聚类中心计算向量的欧氏距离
            # print(type(data[i][0:4:1][0]))--注释测试打印 --2019/4/17 zhangyue

            # if type(data[i][0:4:1][0]) is type("str"):
            #     print(i,j)
            #     print(data[i][0:4:1])
            curr_dist = distance.euclidean(data[i][0:4:1], centers[j][0:4:1])
            if j == 0:
                dist = curr_dist
                cluster = 0
                # print("簇数据:{}".format(dist)) --2019/4/17 zhangyue
            if curr_dist < dist:
                dist = curr_dist
                cluster = j
                # 如果特征向量第一次调用这个方法，对特征向量附加一个簇索引
        if c:
            data[i].append(cluster)
            # 如果特征向量不是第一次调用这个方法，则用新的簇索引替换旧的簇索引。
        else:
            data[i][-1] = cluster
    return data

# 函数来计算簇的平均中心
def calculate_centers(data, k):
    data_length = len(data)
    new_centers = []
    # 迭代簇
    for i in range(0, k):
        count = 0
        total = [0.0, 0.0, 0.0, 0.0]
        for j in range(0, data_length):
            # 如果数据点的簇号与当前迭代的簇号相同，请将其添加到总数中。
            if data[j][5] == i:
                count += 1
                total =list(map(add, data[j][0:4:1], total))
                # 计算平均值
        for l in range(0, 4):
            total[l] = total[l]/count
        new_centers.append(total)
    return new_centers

# 函数计算到平均聚类中心的最近数据点（medoids）
def calculate_medoids(data, centers, k):
    medoids = []
    data_length = len(data)
    for i in range(k):
        closest = 0
        firstFlag = True
        for j in range(data_length):
            if firstFlag:
                closest = j
                curr_dist = distance.euclidean(data[j][0:4:1], centers[i][0:4:1])
                firstFlag = False
            else:
                if distance.euclidean(data[j][0:4:1], centers[i][0:4:1]) < curr_dist:
                    closest = j
                    curr_dist = distance.euclidean(data[j][0:4:1], centers[i][0:4:1])
        medoids.append(data[closest][0:4:1])
    return medoids

# 计算簇中的项数
def count_cluster_items(data, k, c = 5):
    count = numpy.zeros(k)
    data_length = len(data)
    for i in range(k):
        for j in range(data_length):
            if data[j][c] == i:
                count[i] += 1
    return count

# 函数来检查簇是否都是非零的
def check_count(count, k):
    for i in range(0, k):
        if count[i] == 0:
            return False
    return True


# 比较两个函数列表
def compare_counts(old_count, new_count, k):
    for i in range(0, k):
        if old_count[i] != new_count[i]:
            return False
        else:
            continue
    return True  