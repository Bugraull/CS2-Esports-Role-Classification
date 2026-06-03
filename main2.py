import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. HAM VERİYİ YÜKLE
# ==========================================
df = pd.read_csv("cs2_deep_stats_turnuva2.csv")

# ==========================================
# 2. ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING)
# ==========================================
df['Impact_Score'] = df['Rating'] * df['K/D']
df['Sacrifice_Index'] = df['Assists_per_round'] * df['Support_rounds']
df['Star_Factor'] = df['Rating'] / (df['Support_rounds'] + 0.1)

# ==========================================
# 3. K-MEANS MODELLEMESİ (K=3)
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
# 4. KÜMELERE OTOMATİK ROL ATAMA MANTIĞI (KURŞUNGEÇİRMEZ YÖNTEM)
# ==========================================
centers = df.groupby('Cluster')[features].mean()

# 1. Kümeleri Etki Skoru'na göre büyükten küçüğe sırala
centers_sorted_by_impact = centers.sort_values(by='Impact_Score', ascending=False)

# 2. Etki Skoru en yüksek olan küme tartışmasız Yıldız'dır (ZywOo'nun kümesi)
star_cluster = centers_sorted_by_impact.index[0]

# 3. Geriye kalan iki kümeyi al
remaining = centers_sorted_by_impact.index[1:]

# 4. Kalan iki küme arasından, Fedakarlık Skoru daha yüksek olan IGL'dir
if centers.loc[remaining[0], 'Sacrifice_Index'] > centers.loc[remaining[1], 'Sacrifice_Index']:
    igl_cluster = remaining[0]
    entry_cluster = remaining[1]
else:
    igl_cluster = remaining[1]
    entry_cluster = remaining[0]
    
# 5. Doğru isim etiketlerini doğru kümelere yapıştır
role_dict = {
    star_cluster: "Yildiz (Star Rifler)",
    igl_cluster: "Lider / Destek (IGL)",
    entry_cluster: "Tüfekçi / Giriş (Entry)"
}

df['Predicted_Role'] = df['Cluster'].map(role_dict)

# ==========================================
# 5. PROFESYONEL GÖRSELLEŞTİRME (Dinamik Etiketleme)
# ==========================================
plt.figure(figsize=(14, 8))
sns.scatterplot(data=df, x='Impact_Score', y='Sacrifice_Index', hue='Predicted_Role', s=120, palette='Set1', alpha=0.8)

# 1. Yıldız (Star Rifler) kümesinden Rating'i en yüksek 15 oyuncuyu seç
top_15_stars = df[df['Predicted_Role'] == 'Yıldız (Star Rifler)'].sort_values(by='Rating', ascending=False).head(15)

# 2. Lider / Destek (IGL) kümesinden Rating'i en yüksek 5 oyuncuyu seç
top_5_igls = df[df['Predicted_Role'] == 'Lider / Destek (IGL)'].sort_values(by='Rating', ascending=False).head(5)

# 3. İki listeyi birleştir
players_to_label = pd.concat([top_15_stars, top_5_igls])

# 4. Sadece bu seçilmiş 20 özel oyuncunun ismini grafiğe yazdır
for i in players_to_label.index:
    plt.text(df.Impact_Score[i] + 0.015, df.Sacrifice_Index[i] + 0.002, 
             df.Player[i], 
             fontsize=8.5,      # Fontu çok hafif büyüttük
             weight='bold',     # İsimler daha net okunsun diye bold yaptık
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
df.to_csv("cs2_final_roles_advanced_turnuva2.csv", index=False)
print("\nGelişmiş 3'lü analiz tamamlandi! 'cs2_final_roles_advanced.csv' dosyasini ve grafiği inceleyebilirsin.")