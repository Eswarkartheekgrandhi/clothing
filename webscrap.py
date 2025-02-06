import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get("https://www.flipkart.com/search?q=shirts")

time.sleep(5)
images = driver.find_elements(By.TAG_NAME, "img")

for i, img in enumerate(images[:10]): 
    src = img.get_attribute("src")
    if src:
        with open(f"flipkart_shirt_{i}.jpg", "wb") as f:
            f.write(requests.get(src).content)

driver.quit()
