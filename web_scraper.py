import requests
from bs4 import BeautifulSoup  
from selenium import webdriver
import time
import re
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions

chrome_options = Options()
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
chrome_options.add_experimental_option("detach", True)

driver = webdriver.Chrome(executable_path=r"C:\Users\Akshit Tayade\Desktop\BIA 660 Web Mining\chromedriver.exe", chrome_options=chrome_options, options=options)
driver.maximize_window()
driver.get('https://www.gutenberg.org/ebooks/search/?query=&submit_search=Search&sort_order=downloads')
time.sleep(1)

for i in range(4, 29):
    try:
        book = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul/li[{}]'.format(i))
        #opening the link of the book
        link_of_book = book.find_element(By.CSS_SELECTOR, 'a').click()
        time.sleep(2)
        # getting the name of the book
        title_of_book = driver.find_element(By.CSS_SELECTOR, 'h1').text
        time.sleep(2)
        # downloading the txt file
        download_link = driver.find_element(By.XPATH, '//a[text()="Plain Text UTF-8"]').click()
        time.sleep(2)
        # save into txt file
        book_content = driver.find_element(By.CSS_SELECTOR, 'pre') 
        with open('Dataset/'+str(title_of_book)+'.txt', 'w', encoding='utf-8') as f:
            f.write(book_content.text)
        # back to the home page
        time.sleep(2)
        driver.back()
        time.sleep(2)
        driver.back()
        time.sleep(2)
    except:
        pass

next_page = driver.find_element(By.XPATH, '//a[text()="Next"]').click()

for i in range(2, 27):
    try:
        book = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/ul/li[{}]'.format(i))
        #opening the link of the book
        link_of_book = book.find_element(By.CSS_SELECTOR, 'a').click()
        time.sleep(2)
        # getting the name of the book
        title_of_book = driver.find_element(By.CSS_SELECTOR, 'h1').text
        time.sleep(2)
        # downloading the txt file
        download_link = driver.find_element(By.XPATH, '//a[text()="Plain Text UTF-8"]').click()
        time.sleep(2)
        # save into txt file
        book_content = driver.find_element(By.CSS_SELECTOR, 'pre') 
        with open('Dataset/'+str(title_of_book)+'.txt', 'w', encoding='utf-8') as f:
            f.write(book_content.text)
        # back to the home page
        time.sleep(2)
        driver.back()
        time.sleep(2)
        driver.back()
        time.sleep(2)
    except:
        pass

driver.quit()