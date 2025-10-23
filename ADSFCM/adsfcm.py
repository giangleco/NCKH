import time
import numpy as np
import logging
from utility import extract_labels, extract_clusters
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FCM.fcm import Dfcm

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Adsfcm:
    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5,
                 alpha: float = 2, beta: float = 3, max_iter: int = 1000,
                 index: int = 0, metric: str = 'euclidean', accelerated: bool = False):
        if m <= 1:
            raise RuntimeError("m phải > 1")
        self._n_clusters = n_clusters
        self._m = m
        self._epsilon = epsilon
        self._max_iter = max_iter
        self._metric = metric

        self.alpha = alpha
        self.beta = beta
        self.accelerated = accelerated

        self.local_data = None
        self.membership = None
        self.centroids = None
        self.step = 0
        self.process_time = 0

        self.__index = index
        self.__exited = False

    # ===========================
    # Các property
    # ===========================
    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def exited(self):
        return self.__exited

    @property
    def version(self):
        return "1.3"

    @property
    def index(self):
        return self.__index

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def extract_labels(self):
        return extract_labels(membership=self.membership)

    @property
    def extract_clusters(self, labels=None):
        if labels is None:
            labels = self.extract_labels
        return extract_clusters(data=self.local_data, labels=labels, n_clusters=self._n_clusters)

    # ===========================
    # Khởi tạo
    # ===========================

    #Khởi tạo tâm cụm bằng Kmeans++
    def _init_centroid_kmeanspp(self, seed=0):
        if seed > 0:
            np.random.seed(seed)
        n_samples = len(self.local_data)
        logger.info(f"Khởi tạo K-Means++ với n_samples={n_samples}, n_clusters={self._n_clusters}")

        first_idx = np.random.randint(0, n_samples)
        centers = [self.local_data[first_idx]]
        logger.debug(f"Điểm khởi tạo đầu tiên: index={first_idx}")

        for _ in range(1, self._n_clusters):
            dist_sq = np.array([min(np.linalg.norm(x - c) ** 2 for c in centers) for x in self.local_data])
            probs = dist_sq / dist_sq.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            for idx, p in enumerate(cumprobs):
                if r < p:
                    centers.append(self.local_data[idx])
                    logger.debug(f"Thêm centroid mới tại index={idx}")
                    break

        return np.array(centers)

    #Khởi tạo ma trân thành viên
    def _init_membership_semi_supervised(self, f, b, seed=0):
        if seed > 0:
            np.random.seed(seed)
        n_samples = len(self.local_data)
        logger.info(f"Khởi tạo membership semi-supervised với n_samples={n_samples}")

        U0 = np.zeros((n_samples, self._n_clusters))
        for i in range(n_samples):
            if b[i] == 1:
                U0[i] = f[i]
                logger.debug(f"Sample {i} là labeled, sử dụng f[i]={f[i]}")
            else:
                rand_vec = np.random.rand(self._n_clusters)
                U0[i] = rand_vec / rand_vec.sum()
                logger.debug(f"Sample {i} là unlabeled, gán ngẫu nhiên {U0[i]}")
        return U0

    # ===========================
    # Update membership (23)
    # ===========================
    def update_membership_adsfcm(self, X, V, f, b):
        n, d = X.shape
        c = V.shape[0]
        logger.info(f"Cập nhật membership với n={n}, c={c}")
        U = np.zeros((n, c))
        eps = 1e-12
        power = 2 / (self._m - 1)

        dists = np.linalg.norm(X[:, None] - V[None, :], axis=2) ** 2 + eps
        logger.debug(f"Khoảng cách dists shape={dists.shape}")

        for i in range(n):
            ratios = (dists[i]) ** (-power)
            base_membership = ratios / np.sum(ratios)
            denom_factor = self.beta + self.alpha * (1 + b[i]) ** 2
            sum_fi = np.sum(f[i])

            term1 = (self.beta + self.alpha * (1 + b[i]) ** 2
                     - 3 * self.alpha * b[i] * (1 + b[i]) * sum_fi) * base_membership
            term2 = (3 * self.alpha * b[i] * (1 + b[i]) / (denom_factor + eps)) * f[i]
            U[i] = (1.0 / denom_factor) * (term1 + term2)

            U[i] = np.maximum(U[i], 0)
            if U[i].sum() > 0:
                U[i] /= U[i].sum()
            else:
                U[i] = np.ones(c) / c
            logger.debug(f"Sample {i}, U[i]={U[i]}, sum={U[i].sum()}")
        return U

    # ===========================
    # Update centroids (24)
    # ===========================
    def update_centroids_adsfcm(self, X, U, f, b):
        n, d = X.shape
        c = U.shape[1]
        logger.info(f"Cập nhật centroids với n={n}, c={c}")
        V = np.zeros((c, d))
        eps = 1e-12

        for j in range(c):
            weights = self.beta * (U[:, j] ** self._m) + self.alpha * ((U[:, j] * (1 + b) - 3 * b * f[:, j]) ** 2)
            num = np.sum(weights[:, None] * X, axis=0)
            den = np.sum(weights)
            V[j] = num / (den + eps)
            logger.debug(f"Centroid {j} cập nhật: V[j]={V[j]}, den={den}")
        return V

    # ===========================
    # Fit
    # ===========================
    def fit(self, X, f, b, seed=0):
        self.local_data = X
        start_time = time.time()
        logger.info(f"Bắt đầu fit với shape X={X.shape}, n_clusters={self._n_clusters}")

        self.centroids = self._init_centroid_kmeanspp(seed=seed)
        self.membership = self._init_membership_semi_supervised(f=f, b=b, seed=seed)
        logger.info(f"Khởi tạo xong, centroids shape={self.centroids.shape}, membership shape={self.membership.shape}")

        for step in range(self._max_iter):
            old_centroids = self.centroids.copy()
            logger.info(f"Bước {step + 1}/{self._max_iter}")

            self.membership = self.update_membership_adsfcm(X, self.centroids, f, b)
            new_centroids = self.update_centroids_adsfcm(X, self.membership, f, b)
            #Non-Affinity Center Filtering
            if self.accelerated:
                logger.info("Bật chế độ accelerated")
                dists = np.linalg.norm(X[:, None] - old_centroids[None, :], axis=2) ** 2
                logger.debug(f"Khoảng cách dists shape={dists.shape}")
                min_dists = np.min(dists, axis=1)
                nearest = np.argmin(dists, axis=1)
                deltas = np.linalg.norm(new_centroids - old_centroids, axis=1)
                logger.debug(f"min_dists={min_dists[:5]}, nearest={nearest[:5]}, deltas={deltas}")


                u_hat = self.membership.copy()
                for i in range(len(X)):
                    non_affinity = np.where(dists[i] - min_dists[i] >= 0.5 * deltas[nearest[i]])[0]  # Giảm ngưỡng
                    logger.debug(f"Sample {i}, non_affinity centers={non_affinity}")
                    for j in non_affinity:
                        if b[i] == 1:
                            u_hat[i, j] = f[i, j]
                            logger.debug(f"Sample {i}, center {j} là labeled, giữ f[i,j]={f[i, j]}")
                        else:
                            u_hat[i, j] = 0
                            logger.debug(f"Sample {i}, center {j} là non-affinity, đặt 0")

                # Renormalize (Membership Scaling)
                sums = np.sum(u_hat, axis=1)
                logger.debug(f"Sums trước scale={sums[:5]}")
                self.membership = np.where(sums[:, None] > 0, u_hat / sums[:, None], 1.0 / self._n_clusters)
                logger.debug(f"Sums sau scale={np.sum(self.membership, axis=1)[:5]}")

                # Reupdate centroids
                new_centroids = self.update_centroids_adsfcm(X, self.membership, f, b)

            self.centroids = new_centroids

            diff = np.linalg.norm(self.centroids - old_centroids)
            logger.info(f"Diff={diff:.6f}, epsilon={self._epsilon}")
            if diff < self._epsilon:
                self.__exited = True
                logger.info(f"Hội tụ tại bước {step + 1}")
                break

        self.step = step + 1
        self.process_time = time.time() - start_time
        logger.info(f"Kết thúc fit, process_time={self.process_time:.2f}s, steps={self.step}")
        return self.membership, self.centroids, self.step


# ===========================
# Hàm tạo f và b
# ===========================
def build_f_b(Y, n_clusters, labeled_ratio=0.1, seed=42):
    np.random.seed(seed)
    n = len(Y)
    logger.info(f"Tạo f và b với n={n}, n_clusters={n_clusters}, labeled_ratio={labeled_ratio}")

    Y = np.array(Y)
    if not np.issubdtype(Y.dtype, np.integer):
        from sklearn.preprocessing import LabelEncoder
        Y = LabelEncoder().fit_transform(Y)

    f = np.zeros((n, n_clusters))
    b = np.zeros(n)

    n_labeled = int(n * labeled_ratio)
    labeled_idx = np.random.choice(n, size=n_labeled, replace=False)
    logger.debug(f"Chỉ số labeled: {labeled_idx}")

    for i in labeled_idx:
        label = int(Y[i])
        f[i, label] = 1.0
        b[i] = 1
        logger.debug(f"Sample {i}, label={label}, f[i]={f[i]}, b[i]={b[i]}")
        print(f[i])
    return f, b


if __name__ == '__main__':
    from utility import round_float, extract_labels
    from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
    from validity import dunn, silhouette, hypervolume, classification_entropy, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie, accuracy_score

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 1.8  # Thử tối ưu m
    SEED = 42
    SPLIT = '\t'

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report(alg: str, process_time: float, step: int,
                     X: np.ndarray, V: np.ndarray, U: np.ndarray, Y_true: np.ndarray) -> str:
        pred_labels = extract_labels(U)
        # Tối ưu hóa ánh xạ với Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(-np.histogram2d(Y_true, pred_labels, bins=n_clusters)[0])
        mapped_labels = np.zeros_like(pred_labels)
        for true_label, pred_label in zip(col_ind, row_ind):
            mapped_labels[pred_labels == pred_label] = true_label
        ac = accuracy_score(Y_true, mapped_labels)
        labels = extract_labels(U)  # Giữ nguyên cho các chỉ số khác
        results = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),
            wdvl(davies_bouldin(X, labels)),
            wdvl(partition_coefficient(U)),
            wdvl(partition_entropy(U)),
            wdvl(Xie_Benie(X, V, U)),
            wdvl(classification_entropy(U)),
            wdvl(silhouette(X, labels)),
            wdvl(hypervolume(U)),
            wdvl(ac, n=3)
        ]
        return SPLIT.join(results)

    data_id = 53
    if data_id in TEST_CASES:
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print("Không thể lấy dữ liệu")
            exit()
        X, Y = _dt['X'], _dt['Y']
        n_clusters = _TEST['n_cluster']
        logger.info(f"Đọc dữ liệu UCI, X shape={X.shape}, n_clusters={n_clusters}")

        dlec = LabelEncoder()
        labels = dlec.fit_transform(Y.ravel())
        logger.info("Encode nhãn hoàn tất")

        f, b = build_f_b(labels, n_clusters=n_clusters, labeled_ratio=0.3, seed=SEED)  # Tăng labeled_ratio

        # Train FCM trước
        fcm = Dfcm(n_clusters=n_clusters, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        U_fcm, V_fcm, steps_fcm = fcm.fit(X, seed=SEED)

        # Khởi tạo ADSFCM từ FCM
        adsfcm = Adsfcm(n_clusters=n_clusters, m=M, alpha=1.5, beta=2.5, epsilon=EPSILON, max_iter=MAX_ITER, accelerated=True)
        adsfcm.membership = U_fcm.copy()
        adsfcm.centroids = V_fcm.copy()
        logger.info("Bắt đầu huấn luyện ADSFCM với khởi tạo từ FCM")
        U, V, steps = adsfcm.fit(X, f, b, seed=SEED)

        # Visualize Membership Matrix
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 hàng, 2 cột

        # --- Hình 1: Heatmap Membership Matrix ---
        im = axs[0].imshow(U, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=axs[0], label='Degree of Membership')
        axs[0].set_title('Ma trận thành viên Heatmap ADSFCM')
        axs[0].set_xlabel('Chỉ số cụm')
        axs[0].set_ylabel('Chỉ số mẫu')

        # --- Hình 2: Scatter Plot với Centroids ---
        scatter = axs[1].scatter(X[:, 0], X[:, 1], c=np.argmax(U, axis=1), cmap='viridis')
        axs[1].scatter(V[:, 0], V[:, 1], c='red', marker='x', s=200, label='Centroids')
        axs[1].set_title('Dữ liệu điểm và tâm cụm thuật toán ADSFCM')
        axs[1].set_xlabel('Feature 1')
        axs[1].set_ylabel('Feature 2')
        fig.colorbar(scatter, ax=axs[1], label='Assigned Cluster')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

        titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'PE-', 'XB-', 'CE-', 'SI+', 'FH+', 'AC+']
        print(SPLIT.join(titles))
        print(write_report('FCM', fcm.process_time, fcm.step, X, V_fcm, U_fcm, labels))
        print(write_report('ADSFCM', adsfcm.process_time, adsfcm.step, X, V, U, labels))