import numpy as np
import pandas as pd
from FCM import *
class SSFCM(FCM):

    def __init__(self, c, m=2, max_iter=1000, epsilon=1e-5):
        super().__init__(c=c, m=m, max_iter=max_iter, epsilon=epsilon)
    
    def init_u_bar(self,labels):
        #khởi tạo mảng 0 có shape NxC
        self.u_bar=np.zeros((len(labels), self.c))
        #xét từng phần tử có nhãn thì gán tương ứng điểm dữ liệu cho tâm cụm = 1
        for index, val in enumerate(labels):
            if val!=-1:
                self.u_bar[index][val]=1

        return self.u_bar

    def update_membership(self,data):
        dki=1/(np.linalg.norm(data[:, np.newaxis, :]-self.tamcum[np.newaxis, :, :], axis=2)**(2/(self.m-1)))
        dkj=np.sum(dki, axis=1, keepdims=True)
        return self.u_bar + (1-np.sum(self.u_bar, axis=1, keepdims=True))*(dki/dkj)
    
    def calculate_v(self,data):
        temp=(self.U-self.u_bar).T ** self.m #CxN
        TS=np.dot(temp, data)
        MS=division_by_zero(np.sum(temp, axis=1, keepdims=True))
        return TS/MS

    def fit(self,data,labels):
        self.u_bar=self.init_u_bar(labels=labels)
        super().fit(data)

def init_semi_data(labels, ratio):
    # ratio: Tỉ lệ nhãn
    # Chuyển đổi nhãn từ String -> int
    convert_label = np.unique(labels)
    label_to_index = {label:index for index,label in enumerate(convert_label)}
    labels = np.array([label_to_index[label] for label in labels])

    # Lấy số lượng điểm dữ liệu không làm dữ liệu bán gíam sát
    n_not_semi = int(len(labels) * (1-ratio))

    np.random.seed(42)
    # Chọn ra n_semi điểm ngẫu nhiên chuyển label 
    unlabel = np.random.choice(a=[i for i in range(len(labels))], size=n_not_semi,replace=False)

    # Chuyển các giá trị thành -1
    for index in range(len(labels)):
        if index in unlabel:
            labels[index] = -1

    return labels


if __name__ =='__main__':

    df = pd.read_csv('data.csv')
    
    labels = df['Class'].values
    semi_labels = init_semi_data(labels, 0.7)
    df['Class'] = semi_labels

    data = df.iloc[:, :-1].values
    n_cluster = len(np.unique(labels))  
    ssfcm = SSFCM(c=n_cluster, m=2, max_iter=1000, epsilon=1e-5)
   
    ssfcm.fit(data, labels=semi_labels)

    print("u_bar matrix:")
    print(ssfcm.u_bar)

    print('Ma tran thanh vien:')
    print(ssfcm.U)
    
    print("tam cum:")
    print(ssfcm.tamcum)

    print('Labels:')
    print(ssfcm.predict())

    class_column=df['Class']
    #lấy nhãn về 3 loài hoa
    unique_classes=np.unique(class_column)
    #gán loài 1: 0, loài 2: 1, loài 3: 2
    class_indices = np.searchsorted(unique_classes, class_column)

    import time
    from utility import round_float, extract_labels
    from validity import dunn, davies_bouldin, partition_coefficient, Xie_Benie, silhouette, classification_entropy, hypervolume, f1_score, accuracy_score

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    SPLIT = '\t'
    a=np.e
    average='weighted'
    y_pre=ssfcm.predict()
    y_true=class_indices
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
            wdvl(classification_entropy(U,a)), #CE-
            wdvl(silhouette(X, labels)), #SI+
            wdvl(hypervolume(U,M)), #FHV-
            wdvl(Xie_Benie(X, V, U)),  # XB
            wdvl(f1_score(y_true, y_pre, average )), #F1+
            wdvl(accuracy_score(y_true, y_pre)), #AC+
        ]
        return SPLIT.join(kqdg)


    titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'CE-', 'SI+', 'FHV-', 'XB-', 'F1+', 'AC+']
    print(SPLIT.join(titles))
    print(write_report_fcm(alg='SSFCM', index=0, process_time=ssfcm.process_time, step=ssfcm.i, X=data, V=ssfcm.tamcum, U=ssfcm.U))
