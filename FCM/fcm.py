import time
import numpy as np 
from utility import distance_cdist, extract_labels, extract_clusters

class Dfcm():
    def __init__(self, n_clusters: int, m: float = 2, epsilon: float = 1e-5, max_iter: int = 10000, index: int = 0, metric: str = 'euclidean'):  # euclidean|chebyshev
        if m <= 1:
            raise RuntimeError('m>1')
        self._metric = metric
        self._n_clusters = n_clusters
        self._m = m
        self._epsilon = epsilon
        self._max_iter = max_iter

        self.local_data = None
        self.membership = None
        self.centroids = None

        self.process_time = 0
        self.step = 0

        self.__exited = False
        self.__index = index

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def exited(self) -> bool:
        return self.__exited

    @property
    def version(self) -> str:
        return '1.3'

    @exited.setter
    def exited(self, value: bool):
        self.__exited = value

    @property
    def index(self) -> int:
        return self.__index

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def extract_labels(self) -> np.ndarray:
        return extract_labels(membership=self.membership)

    @property
    def extract_clusters(self, labels: np.ndarray = None) -> list:
        if labels is None:
            labels = self.extract_labels
        return extract_clusters(data=self.local_data, labels=labels, n_clusters=self._n_clusters)

    # Dự đoán 1 điểm mới thuộc nhãn nào
    def predict(self, new_data: np.ndarray, m: float = 2) -> np.ndarray:
        _new_u = self.update_membership(new_data, self.centroids, m=m)
        return extract_labels(membership=_new_u)

    def compute_j(self, data: np.ndarray) -> float:
        _distance = distance_cdist(data, self.centroids, metric=self._metric)
        return np.sum((self.membership ** self._m) * (_distance ** 2))

    @staticmethod
    def _division_by_zero(data: np.ndarray | float) -> np.ndarray | float:
        if isinstance(data, np.ndarray):
            data[data == 0] = np.finfo(float).eps
            return data
        return np.finfo(float).eps if data == 0 else data

    # INIT CENTROID BEGIN ==============================================
    def _init_centroid_random(self, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed=seed)
        return self.local_data[np.random.choice(len(self.local_data), self._n_clusters, replace=False)]
    # INIT CENTROID END ================================================

    # INIT MEMBERSHIP BEGIN ============================================
    # Khởi tạo ma trận thành viên theo phương pháp ngẫu nhiên
    def _init_membership_random(self, seed: int = 0) -> np.ndarray:
        if seed > 0:
            np.random.seed(seed=seed)
        n_samples = len(self.local_data)
        U0 = np.random.rand(n_samples, self._n_clusters)
        return U0 / U0.sum(axis=1)[:, None]

    # INIT MEMBERSHIP END ================================================
    def _update_centroids(self, data: np.ndarray, membership: np.ndarray) -> np.ndarray:  # Cập nhật ma trận tâm cụm
        _um = membership ** self._m  # (N, C)
        numerator = np.dot(_um.T, data)  # (C, N) x (N, D) = (C, D)
        denominator = _um.sum(axis=0)[:, np.newaxis]  # (C, 1)
        return numerator / self._division_by_zero(denominator)

    @staticmethod
    def calculate_membership_by_distances(distances: np.ndarray, m: float = 2) -> np.ndarray:
        _d = distances[:, :, None] * ((1 / Dfcm._division_by_zero(distances))[:, None, :])
        power = 2 / (m - 1)
        mau = (_d ** power).sum(axis=2)
        return 1 / mau

    def calculate_membership(self, distances: np.ndarray, m: float = 2) -> np.ndarray:  # Cập nhật ma trận độ thuộc
        return self.calculate_membership_by_distances(distances=distances, m=m)

    # CHECK EXIT BEGIN ================================================
    def update_membership(self, data: np.ndarray, centroids: np.ndarray, m: float = 2) -> np.ndarray:
        _distances = distance_cdist(data, centroids, metric=self._metric)  # Khoảng cách giữa data và centroids
        return self.calculate_membership(distances=_distances, m=m)

    def __max_abs_epsilon(self, val: np.ndarray) -> bool:
        if not self.__exited:
            self.__exited = (np.abs(val)).max(axis=(0, 1)) < self._epsilon
        return self.__exited

    def check_exit_by_membership(self, membership: np.ndarray) -> bool:
        return self.__max_abs_epsilon(self.membership - membership)

    def check_exit_by_centroids(self, centroids: np.ndarray) -> bool:
        return self.__max_abs_epsilon(self.centroids - centroids)
    # CHECK EXIT END ================================================

    # FIT BEGIN ==============================================
    def __fit_with_centroid(self, init_v: np.ndarray = None, seed: int = 0, device: str = 'CPU'):
        self.centroids = self._init_centroid_random(seed=seed) if init_v is None else init_v
        for _step in range(self._max_iter):
            old_v = self.centroids.copy()
            self.membership = self.update_membership(self.local_data, old_v, m=self._m)
            self.centroids = self._update_centroids(self.local_data, self.membership)
            if self.check_exit_by_centroids(old_v):
                break
        self.step = _step + 1

    def __fit_with_membership(self, init_u: np.ndarray = None, seed: int = 0, device: str = 'CPU'):
        self.membership = self._init_membership_random(seed=seed) if init_u is None else init_u
        for _step in range(self._max_iter):
            old_u = self.membership.copy()
            self.centroids = self._update_centroids(self.local_data, old_u)
            self.membership = self.update_membership(self.local_data, self.centroids, m=self._m)
            if self.check_exit_by_membership(old_u):
                break
        self.step = _step + 1

    def fit(self, data: np.ndarray, init_u: np.ndarray = None, init_v: np.ndarray = None, seed: int = 0, with_u: bool = True, device: str = 'CPU') -> tuple:
        self.local_data = data
        _start_tm = time.time()
        if with_u or init_u:
            self.__fit_with_membership(init_u=init_u, seed=seed, device=device)
        else:
            self.__fit_with_centroid(init_v=init_v, seed=seed, device=device)
        # -----------------------------------------------
        self.process_time = time.time() - _start_tm
        return self.membership, self.centroids, self.step
    # FIT END ==============================================

if __name__ == '__main__':
    import time
    from utility import round_float, extract_labels
    from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
    from validity import dunn, davies_bouldin, partition_coefficient, partition_entropy, Xie_Benie

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report_fcm(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(partition_entropy(U)),  # PE
            wdvl(Xie_Benie(X, V, U)),  # XB
        ]
        return SPLIT.join(kqdg)
    # =======================================

    clustering_report = []
    data_id = 53
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        n_clusters = _TEST['n_cluster']
        # ===============================================
        dlec = LabelEncoder()
        labels = dlec.fit_transform(_dt['Y'].flatten())
        

        fcm = Dfcm(n_clusters=n_clusters, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        fcm.fit(data=X, seed=SEED)
        # --------------------------------  
        titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'PE-', 'XB-']
        print(SPLIT.join(titles))
        print(write_report_fcm(alg='FCM', index=0, process_time=fcm.process_time, step=fcm.step, X=X, V=fcm.centroids, U=fcm.membership))

