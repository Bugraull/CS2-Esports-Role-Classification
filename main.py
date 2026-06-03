import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. HAM VERİYİ YÜKLE
# ==========================================
# DİKKAT: Hangi turnuvayı çizdiriyorsan buradaki CSV ismini ona göre değiştir!
df = pd.read_csv("cs2_deep_stats.csv") 

# ==========================================
# 2. ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
# ==========================================
df['Impact_Score'] = df['Rating'] * df['K/D']
df['Sacrifice_Index'] = df['Assists_per_round'] * df['Support_rounds']
df['Star_Factor'] = df['Rating'] / (df['Support_rounds'] + 0.1)

# ==========================================
# 3. K-MEANS MODELLEMESİ (Kesinlikle K=3)
# ==========================================
features = ['Rating', 'K/D', 'Assists_per_round', 'Support_rounds', 
            'Impact_Score', 'Sacrifice_Index', 'Star_Factor']

x = df[features]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Modeli kur ve 3 ana role ayır
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(x_scaled)

# ==========================================
# 4. KÜMELERE OTOMATİK ROL ATAMA MANTIĞI (SABİT NUMARALANDIRMA)
# ==========================================
centers = df.groupby('Cluster')[features].mean()

# 1. Kümeleri Etki Skoru'na göre büyükten küçüğe sırala
centers_sorted_by_impact = centers.sort_values(by='Impact_Score', ascending=False)

# 2. En yüksek etki = Yıldız
star_cluster = centers_sorted_by_impact.index[0]

# 3. Kalan ikisini al
remaining = centers_sorted_by_impact.index[1:]

# 4. Kalanlardan Fedakarlığı yüksek olan = IGL, diğeri = Entry
if centers.loc[remaining[0], 'Sacrifice_Index'] > centers.loc[remaining[1], 'Sacrifice_Index']:
    igl_cluster = remaining[0]
    entry_cluster = remaining[1]
else:
    igl_cluster = remaining[1]
    entry_cluster = remaining[0]
    
# 5. Önce doğru rolleri isim olarak atayalım
role_dict = {
    star_cluster: "Yildiz (Star Rifler)",
    igl_cluster: "Lider / Destek (IGL)",
    entry_cluster: "Tüfekçi / Giriş (Entry)"
}
df['Predicted_Role'] = df['Cluster'].map(role_dict)

# 6. KESİN ÇÖZÜM: K-Means'in rastgele verdiği Cluster numaralarını ezip
# senin istediğin standart formata (0=Star, 1=Entry, 2=IGL) zorla dönüştürüyoruz!
sabit_numara_dict = {
    "Yildiz (Star Rifler)": 0,
    "Tüfekçi / Giriş (Entry)": 1,
    "Lider / Destek (IGL)": 2
}
df['Cluster'] = df['Predicted_Role'].map(sabit_numara_dict)

# ==========================================
# 5. PROFESYONEL GÖRSELLEŞTİRME (Her Rolün En İyilerini Etiketle)
# ==========================================
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='Impact_Score', y='Sacrifice_Index', hue='Predicted_Role', s=120, palette='Set1', alpha=0.8)

# Her 3 kümeden de temsilciler seçelim ki grafikte hepsi ismen yer alsın
top_stars = df[df['Predicted_Role'] == 'Yildiz (Star Rifler)'].sort_values(by='Rating', ascending=False).head(12)
top_igls = df[df['Predicted_Role'] == 'Lider / Destek (IGL)'].sort_values(by='Rating', ascending=False).head(5)
top_entries = df[df['Predicted_Role'] == 'Tüfekçi / Giriş (Entry)'].sort_values(by='Rating', ascending=False).head(5)

# Listeleri birleştir
players_to_label = pd.concat([top_stars, top_igls, top_entries])

for i in players_to_label.index:
    plt.text(df.Impact_Score[i] + 0.015, df.Sacrifice_Index[i] + 0.002, 
             df.Player[i], 
             fontsize=8.5, 
             weight='bold', 
             color='#222222',   
             alpha=0.9)

plt.title('CS2 Rol Analizi: Etki Skoru (Impact) vs Fedakarlik Skoru (Sacrifice)', fontsize=14, pad=15)
plt.xlabel('Saf Etki Skoru (Rating * K/D)', fontsize=12)
plt.ylabel('Fedakarlık Skoru (Asist * Support)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 6. SONUÇLARI KAYDET
# ==========================================
df.to_csv("cs2_final_roles_advanced.csv", index=False)