import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. VERİLERİ YÜKLE
df1 = pd.read_csv("cs2_final_roles_advanced.csv") 
df2 = pd.read_csv("cs2_final_roles_advanced_turnuva2.csv")

# ==========================================
# 2. ANALİTİK FİLTRELEME (İSİM BELİRLEME MANTIĞI)
# ==========================================
# A) Ortak Oyuncuları Bul (İki listede de olanlar)
ortak_oyuncular = set(df1['Player']).intersection(set(df2['Player']))

# B) Turnuvalara Özgü (Eşsiz) Oyuncuları Ayır
essiz_df1 = df1[~df1['Player'].isin(ortak_oyuncular)]
essiz_df2 = df2[~df2['Player'].isin(ortak_oyuncular)]

# C) Eşsiz oyuncular arasından, her kümenin (Rolün) en yüksek Impact Score'a sahip olanını bul
top_essiz_1_idx = essiz_df1.groupby('Predicted_Role')['Impact_Score'].idxmax()
top_essiz_1_isimler = essiz_df1.loc[top_essiz_1_idx, 'Player'].tolist()

top_essiz_2_idx = essiz_df2.groupby('Predicted_Role')['Impact_Score'].idxmax()
top_essiz_2_isimler = essiz_df2.loc[top_essiz_2_idx, 'Player'].tolist()

# D) Yazdırılacak Nihai İsim Listelerini Birleştir
etiketlenecek_df1 = list(ortak_oyuncular) + top_essiz_1_isimler
etiketlenecek_df2 = list(ortak_oyuncular) + top_essiz_2_isimler

# ==========================================
# 3. GÖRSELLEŞTİRME
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# --- BİRİNCİ TURNUVA (SOL GRAFİK) ---
sns.scatterplot(data=df1, x='Impact_Score', y='Sacrifice_Index', hue='Predicted_Role', 
                s=120, palette='Set1', alpha=0.8, ax=axes[0], legend=False)
axes[0].set_title('Turnuva 1: IEM Katowice Analizi', fontsize=14, pad=10)
axes[0].set_xlabel('Saf Etki Skoru (Rating * K/D)')
axes[0].set_ylabel('Fedakarlik Skoru (Asist * Support)')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Sadece filtreden geçen isimleri yazdır
for i in df1.index:
    if df1.loc[i, 'Player'] in etiketlenecek_df1:
        # Ortak oyuncuları siyah, küme lideri sürpriz oyuncuları mavimsi yazdırarak farkı gösterelim
        renk = '#222222' if df1.loc[i, 'Player'] in ortak_oyuncular else '#00008B'
        kalinlik = 'semibold' if df1.loc[i, 'Player'] in ortak_oyuncular else 'bold'
        
        axes[0].text(df1.loc[i, 'Impact_Score'] + 0.015, df1.loc[i, 'Sacrifice_Index'] + 0.002, 
                     df1.loc[i, 'Player'], fontsize=8.5, weight=kalinlik, color=renk, alpha=0.9)

# --- İKİNCİ TURNUVA (SAĞ GRAFİK) ---
sns.scatterplot(data=df2, x='Impact_Score', y='Sacrifice_Index', hue='Predicted_Role', 
                s=120, palette='Set1', alpha=0.8, ax=axes[1])
axes[1].set_title('Turnuva 2: Budapest Major     Analizi', fontsize=14, pad=10)
axes[1].set_xlabel('Saf Etki Skoru (Rating * K/D)')
axes[1].set_ylabel('') 
axes[1].grid(True, linestyle='--', alpha=0.5)

# Sadece filtreden geçen isimleri yazdır
for i in df2.index:
    if df2.loc[i, 'Player'] in etiketlenecek_df2:
        renk = '#222222' if df2.loc[i, 'Player'] in ortak_oyuncular else '#00008B'
        kalinlik = 'semibold' if df2.loc[i, 'Player'] in ortak_oyuncular else 'bold'
        
        axes[1].text(df2.loc[i, 'Impact_Score'] + 0.015, df2.loc[i, 'Sacrifice_Index'] + 0.002, 
                     df2.loc[i, 'Player'], fontsize=8.5, weight=kalinlik, color=renk, alpha=0.9)

plt.tight_layout()
plt.show()