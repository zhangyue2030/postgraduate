import kmeans
import stats
import random

# 提供两种划分方法，即课上说的k-平均和k-中心点算法
# k-means :由簇的中心来代表簇；
# k-medoids: 每个簇由簇中的某个数据对象来代表
args = ("means", "medoids")
args = args[0]
# Setting K value.
k = 3
# Setting acceptable initialisation flag.
init_flag = False
converge_flag = False
counter = 0
centers = []
feat_vects = []
new_count = []
clustered_data = []

while not init_flag:
    # feat_vects: list
    # 从数据文件生成向量
    print("\n加载数据中...")
    feat_vects = kmeans.get_feat_vects("iris.data", 5)
    print("初始化簇中心...")

    # 将最接近随机生成点的数据点指定为中心
    centers = random.sample(feat_vects, k)
    if args is "medoids":
        centers = kmeans.calculate_medoids(feat_vects, centers, k)
    # 执行初始聚类
    print("开始处理聚类数据...")
    clustered_data = kmeans.cluster_data(centers, feat_vects, True)
    # 计算每个集群中的成员数
    print("正在计算簇成员数量...")
    new_count = kmeans.count_cluster_items(clustered_data, k)
    print(new_count)
    print("检查可重分配簇...\n")
    init_flag = kmeans.check_count(new_count, k)
    if not init_flag:
        print("\n检测到可重分配簇，正在重新初始化算法.")
    else:
        print("\n集群是可重新分配的，继续迭代")
        # 为while循环设置标志和计数器
    converge_flag = False
    counter = 1

# 在簇未聚合时继续执行
while not bool(converge_flag):
    print("\nIteration", counter)
    old_count = new_count
    if args is "means":
        print("计算新的聚类中心（平均）...")
        centers = kmeans.calculate_centers(clustered_data, k)
    elif args is "medoids":
        print("计算新的簇中心（中心点）...")
        centers = kmeans.calculate_centers(clustered_data, k)
        centers = kmeans.calculate_medoids(clustered_data, centers, k)
    print("正在聚类数据...")
    clustered_data = kmeans.cluster_data(centers, feat_vects, False)
    print("正在计算每个簇的成员...")
    new_count = kmeans.count_cluster_items(clustered_data, k)
    print(new_count)
    # 比较计数以检查收敛性
    converge_flag = kmeans.compare_counts(old_count, new_count, k)
    counter += 1

print("\n计数保持不变，簇已收敛到最优值。")
print( "正在评估簇...")
# 使用简单的指标和DBI评估集群.
stats.evaluate_clusters_simple(clustered_data, new_count, k)
stats.evaluate_clusters_complex(clustered_data, centers, new_count, k)