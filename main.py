from sklearn.datasets import make_blobs
import k_means_clustering as kc
import matplotlib.pyplot as plt
from sklearn import metrics


color = ['b','r','g','c','m','y','k','w']
marker = ['.','o','v','^','<','>',"s","p"]

X, y = make_blobs(n_samples=100,
                  n_features=2,
                  centers=5)

plt.scatter(X[:, 0], X[:, 1],
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()

rand_index = []
rand_index_adjust = []
silhoue = []

for k in range(3,9):

    grouped, logs, it, centers = kc.cluster(X, k,iter_num=10000)

    cluster = 0

    correct_label = []
    predict_label = []

    for p in grouped:
        plt.scatter(p[0],p[1], s=50, c=color[int(p[2])],
            marker=marker[int(p[2])], edgecolor='black',
            )

        predict_label.append(int(p[2]))

        for i in range(100):
            if p[0] == X[i][0] and p[1] == X[i][1]:
                correct_label.append(y[i])

    plt.scatter(centers[:k, 0],centers[:k, 1],
                s=250, marker='*',
                c='red', edgecolor='black')

    label= []
    for i in range(len(grouped)):
        label.append(int(grouped[i][2]))
    print("k = " + str(k) +" / silhouette: "+ str(metrics.silhouette_score(grouped[:,:2],label))
          +" / Rand Index: " + str(metrics.rand_score(predict_label, correct_label))
          +" / Adjust Rand Index" +str(metrics.adjusted_rand_score(predict_label, correct_label)))

    rand_index.append(metrics.rand_score(predict_label, correct_label))
    rand_index_adjust.append(metrics.adjusted_rand_score(predict_label, correct_label))
    silhoue.append(metrics.silhouette_score(grouped[:,:2],label))

    plt.title("K: " + str(k))
    plt.grid()
    plt.tight_layout()
    plt.show()

print("Rand Index 기준 최적 k: "+ str(rand_index.index(max(rand_index))+3) + "  /  " + str(max(rand_index)))
print("Adjust Rand Index 기준 최적 k: "+ str(rand_index_adjust.index(max(rand_index_adjust))+3) + "  /  " + str(max(rand_index_adjust)))
print("silhouette 기준 최적 k: "+ str(silhoue.index(max(silhoue))+3) + "  /  " + str(max(silhoue)))