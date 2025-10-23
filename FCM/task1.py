import cv2
import numpy as np
import sys
import time  # Import thư viện time

IMAGE_PATH = 'dataset/Flood_Area_Segmentation/Image/0.jpg'
LABEL_PATH = 'dataset/Flood_Area_Segmentation/Mask/0.png' 
ROUND_FLOAT = 3
SPLIT = '\t' 
# --- 1. IMPORT CÁC TỆP CỦA BẠN ---
try:
    from fcm import Dfcm
except ImportError:
    print("LỖI: Không tìm thấy tệp 'fcm.py'. Hãy đảm bảo nó ở cùng thư mục.")
    sys.exit()

try:
    # Import tất cả các hàm chỉ số từ tệp validity.py
    from validity import (
        dunn, 
        davies_bouldin, 
        silhouette,
        partition_coefficient,
        classification_entropy,
        Xie_Benie,
        accuracy_score, 
        f1_score,
        partition_entropy
    )
    from utility import extract_labels
except ImportError:
    print("LỖI: Không tìm thấy tệp 'validity.py' hoặc 'utility.py'.")
    sys.exit()
def round_float(number: float, n: int = 3) -> float:
    """Hàm làm tròn số (Lấy từ utility.py)"""
    if n == 0:
        return int(number)
    return round(number, n)

def load_and_prepare_data(image_path, label_path, water_label_id):
    print("Đang tải và chuẩn bị dữ liệu!")
    load_time_start = time.time()

    # Tải ảnh gốc
    img = cv2.imread(image_path)
    if img is None:
        print(f"LỖI: Không thể tìm thấy ảnh gốc tại: {image_path}")
        return None, None, None, 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape

    # Chuyển ảnh sang dạng (Số_pixel, 3) và chuẩn hóa [0, 1]
    pixel_data = img_rgb.astype(np.float32) / 255.0
    reshaped_data = pixel_data.reshape(-1, 3) 
    n_samples = reshaped_data.shape[0]

    # Tải ảnh nhãn
    label_img_gray = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label_img_gray is None:
        print(f"LỖI: Không thể tìm thấy ảnh nhãn tại: {label_path}")
        return None, None, None, 0
    
    # Đảm bảo ảnh nhãn có cùng kích thước
    if label_img_gray.shape[0:2] != original_shape[0:2]:
        label_img_gray = cv2.resize(label_img_gray, (original_shape[1], original_shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
    
    labels_true_flat = label_img_gray.reshape(-1) # (N_pixels,)
    labels_true_binary = np.where(labels_true_flat == water_label_id, 1, 0)
    load_time_total = time.time() - load_time_start
    
    return reshaped_data, labels_true_binary, n_samples, load_time_total

def run_fcm_clustering(data, n_clusters, m, epsilon, max_iter, seed):
    print(f"Bắt đầu chạy FCM với k={n_clusters} cụm!")
    fcm = Dfcm(n_clusters=n_clusters, m=m, epsilon=epsilon, max_iter=max_iter)
    
    # Chạy thuật toán
    membership, centroids, steps = fcm.fit(data=data, seed=seed, with_u=True)
    process_time = fcm.process_time # Lấy thời gian xử lý
    
    return membership, centroids, process_time, steps
def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
    return str(round_float(val, n=n))
def write_report_fcm(alg: str, 
                     process_time: float, 
                     step: int, 
                     X: np.ndarray,          # Dữ liệu
                     V: np.ndarray,          # Centroids
                     U: np.ndarray,          # Membership
                     labels_true: np.ndarray, # Nhãn thật (binary)
                     sample_size: int,
                     seed: int) -> str:
    
    # 1. Giải mờ (Lấy nhãn dự đoán)
    labels_pred = extract_labels(U) 

    # 2. Chuẩn bị dữ liệu mẫu (cho các chỉ số chậm)
    n_samples = X.shape[0]
    if n_samples > sample_size:
        np.random.seed(seed)
        sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    else:
        sample_indices = np.arange(n_samples)
    data_sample = X[sample_indices]
    labels_pred_sample = labels_pred[sample_indices]

    try:
        acc_normal = accuracy_score(labels_true, labels_pred)
        labels_pred_inverted = 1 - labels_pred
        acc_inverted = accuracy_score(labels_true, labels_pred_inverted)
        final_acc = max(acc_normal, acc_inverted)
        
        if acc_normal > acc_inverted:
            final_pred_labels = labels_pred
        else:
            final_pred_labels = labels_pred_inverted
    except Exception:
        print("Erro Metrix")

    kqdg = [
        alg,
        wdvl(process_time, n=2),
        str(step),
        wdvl(dunn(data_sample, labels_pred_sample)),  # DI
        wdvl(davies_bouldin(data_sample, labels_pred_sample)),  # DB
        wdvl(partition_coefficient(U)),  # PC
        wdvl(partition_entropy(U)),  # PE
        wdvl(Xie_Benie(X, V, U)),  # XB
        wdvl(accuracy_score(labels_true, labels_pred)), #AC
        wdvl(f1_score(labels_true, final_pred_labels, average='weighted')) #F1
    ]
    return SPLIT.join(kqdg)

def main():

    N_CLUSTERS = 2
    WATER_LABEL_ID = 255  # 255 = Trắng (Nước), 0 = Đen (Không phải Nước)

    M = 2.0
    EPSILON = 1e-5
    MAX_ITER = 1000
    SEED = 42
    SAMPLE_SIZE = 10000 
    
    # --- BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU ---
    data, labels_true, n_samples, load_time = load_and_prepare_data(
        IMAGE_PATH, LABEL_PATH, WATER_LABEL_ID
    )
    
    if data is None: # Xử lý lỗi nếu không tải được tệp
        return

    print(f"Thời gian lấy dữ liệu: {load_time:.3f}")
    print(f"size={n_samples} x {data.shape[1]}") # In ra size giống format

    # --- BƯỚC 2: CHẠY THUẬT TOÁN FCM ---
    membership, centroids, process_time, steps = run_fcm_clustering(
        data, N_CLUSTERS, M, EPSILON, MAX_ITER, SEED
    )

    titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'PE-', 'XB-', 'AC+', 'F1+']
    print(SPLIT.join(titles))
    print(write_report_fcm(
        alg='FCM',
        process_time=process_time,
        step=steps,
        X=data,
        V=centroids,
        U=membership,
        labels_true=labels_true,
        sample_size=SAMPLE_SIZE,
        seed=SEED
    ))
    
if __name__ == "__main__":
    main()