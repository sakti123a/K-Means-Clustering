import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# --- TAMBAHKAN LIBRARY UNTUK SILHOUETTE SCORE ---
from sklearn.metrics import silhouette_score

# LANGKAH A: MEMBACA FILE EXCEL

# 1. Membaca file Excel pada sheet ke-3 (index=2)
file_path = 'Tugas Superstore DataSet.xlsx'
df = pd.read_excel(file_path, sheet_name='RFM')

# Jalankan df.head() untuk memastikan kolomnya sudah benar
print("Data awal dari sheet ke-3:")
print(df.head())

# 2. Memilih fitur yang akan digunakan untuk K-Means
kolom_fitur = ['R Score', 'F Score', 'M Score']
df = df.dropna(subset=kolom_fitur)
X = df[kolom_fitur]

# LANGKAH A: ELBOW METHOD & SILHOUETTE ANALYSIS
# ==========================================
wcss = []
silhouette_scores = []
rentang_k = range(1, 11) # Mencoba cluster 1 sampai 10

for k in rentang_k:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    # Silhouette score hanya bisa dihitung jika jumlah cluster > 1
    if k > 1:
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

# Visualisasi 1: Elbow Method
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1) # Grafik pertama di sebelah kiri
plt.plot(rentang_k, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method untuk Mencari K Terbaik')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS / Inertia')
plt.xticks(rentang_k)
plt.grid(True)

# Visualisasi 2: Silhouette Score Method
plt.subplot(1, 2, 2) # Grafik kedua di sebelah kanan
plt.plot(range(2, 11), silhouette_scores, marker='s', linestyle='--', color='g')
plt.title('Silhouette Score untuk Setiap K')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# LANGKAH B: EKSEKUSI K-MEANS
# ==========================================
# Silakan tinjau kembali nilai 7 ini berdasarkan puncak tertinggi grafik Silhouette Score nanti
jumlah_cluster_terbaik = 4

kmeans_final = KMeans(n_clusters=jumlah_cluster_terbaik, init='k-means++', random_state=42)
df['Cluster'] = kmeans_final.fit_predict(X)

# Hitung Silhouette Score Final untuk cluster terpilih
score_final = silhouette_score(X, kmeans_final.labels_)

# ==========================================
# LANGKAH C: MELIHAT HASIL & PROFILING
# ==========================================
print("\n================ EVALUASI MODEL ================")
print(f"Jumlah Cluster Terpilih: {jumlah_cluster_terbaik}")
print(f"Silhouette Score Model: {score_final:.4f}")
print("================================================")

print("\nHasil Clustering (5 baris pertama):")
print(df[['Customer ID', 'Cluster'] + kolom_fitur].head())

# Analisis Karakteristik tiap Cluster (Profiling)
print("\nRata-rata Nilai Normalisasi Per Cluster:")
profiling = df.groupby('Cluster')[kolom_fitur].mean()
print(profiling)

kamus_cluster = {
    0: 'Loyal Customer',
    1: 'At Risk',
    2: 'Champions',
    3: 'Hibernating'
}

# 2. Buat kolom baru bernama 'Segment' berdasarkan pemetaan kamus di atas
df['Segment'] = df['Cluster'].map(kamus_cluster)
# ----------------─────────────────────────────────────────

# Menampilkan hasil setelah namanya diubah (5 baris pertama)
print("\nHasil Klasterisasi dengan Nama Segmen Baru:")
print(df[['Customer ID', 'Cluster', 'Segment'] + kolom_fitur].head())

# 3. Menyimpan kembali hasilnya ke file Excel Baru
output_file = 'Hasil_Clustering_Superstore.xlsx'
df.to_excel(output_file, index=False)

print(f"\nSelesai! Hasil cluster telah disimpan ke dalam file: {output_file}")

# ==========================================
# LANGKAH D: VISUALISASI HASIL CLUSTERING
# ==========================================

# --- PILIHAN B: VISUALISASI 3D (Menampilkan R, F, dan M Sekaligus) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Membuat scatter plot 3 dimensi
scatter_3d = ax.scatter(df['R Score'], df['M Score'], df['F Score'],
                        c=df['Cluster'], cmap='rainbow', alpha=0.7, s=40, edgecolors='w')

ax.view_init(elev=10, azim=45)

# Memberikan label pada masing-masing sumbu koordinat
ax.set_title('Visualisasi 3D K-Means Clustering (RFM)', fontsize=14)
ax.set_xlabel('Recency Score')
ax.set_ylabel('Monetary Score')
ax.set_zlabel('Frequency Score')

# Menambahkan legenda warna cluster
legend1 = ax.legend(*scatter_3d.legend_elements(), title="Clusters", loc="upper right")
ax.add_artist(legend1)

plt.show()
