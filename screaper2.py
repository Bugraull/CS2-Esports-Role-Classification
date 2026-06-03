import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

def deep_hltv_scraper():
    print("Anti-Bot Korumali Tarayici Başlatiliyor...")
    options = uc.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    
    # Sürüm uyumluluğu için 148'de sabit
    driver = uc.Chrome(options=options, version_main=148)
    
    try:
        url = "https://www.hltv.org/stats/players?event=8243    "
        print("Ana tabloya gidiliyor...")
        time.sleep(1)
        driver.get(url)
        
        wait = WebDriverWait(driver, 60)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "stats-table")))
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        table = soup.find('table', {'class': 'stats-table'})
        if not table:
             raise Exception("Ana istatistik tablosu sayfada bulunamadi!")
             
        rows = table.find('tbody').find_all('tr')
        player_data = []
        for row in rows:
            name_elem = row.find('td', {'class': 'playerCol'})
            team_elem = row.find('td', {'class': 'teamCol'})
            if name_elem and name_elem.find('a'):
                name = name_elem.text.strip()
                link = "https://www.hltv.org" + name_elem.find('a')['href']
                team = team_elem.text.strip() if team_elem and team_elem.text.strip() != "" else "Free Agent"
                player_data.append({'name': name, 'team': team, 'link': link})

        final_data = []
        print(f"{len(player_data)} oyuncu için hizli kazima başliyor...")

        for i, p in enumerate(player_data[:80]):
            try:
                print(f"[{i+1}/80] Veri çekiliyor: {p['name']}")
                driver.get(p['link'])
                time.sleep(2) # Bekleme süresini optimize ettik
                
                detail_soup = BeautifulSoup(driver.page_source, 'html.parser')
                stats_dict = {}
                all_stats = detail_soup.find_all('div', {'class': 'stats-row'})
                
                for s_row in all_stats:
                    spans = s_row.find_all('span')
                    if len(spans) >= 2:
                        label = spans[0].text.strip().lower() 
                        value = spans[-1].text.strip()
                        stats_dict[label] = value

                rating_val, kd_val, assists_val, support_val = "0", "0", "0", "0"

                for key, value in stats_dict.items():
                    if "rating" in key: rating_val = value
                    elif "k/d" in key or "kill / death" in key: kd_val = value
                    elif "assists" in key: assists_val = value
                    elif "saved" in key or "support" in key: support_val = value

                final_data.append({
                    "Player": p['name'],
                    "Team": p['team'],   
                    "Rating": rating_val,
                    "K/D": kd_val,
                    "Assists_per_round": assists_val,
                    "Support_rounds": support_val
                })
                
            except Exception as e:
                print(f"Oyuncu atlaniyor: {e}")
                continue
                
        df = pd.DataFrame(final_data)
        for col in ["Rating", "K/D", "Assists_per_round", "Support_rounds"]:
            df[col] = df[col].astype(str).str.replace('%', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df.to_csv("cs2_deep_stats_turnuva2.csv", index=False)
        print("\nTEMİZ VERİ SETİ HAZIR: 'cs2_deep_stats.csv' başariyla kaydedildi.")

    except Exception as e:
        print(f"Hata: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    deep_hltv_scraper()