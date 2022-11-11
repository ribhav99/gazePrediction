import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

def download_files(link, startswith=None, endswith=None):
    r = requests.get(link)
    if 300 > r.status_code >= 200:
        soup = BeautifulSoup(r.content, 'html.parser')
        for atag in tqdm(soup.find_all('a')):
            url = atag.get('href')

            flag = False
            if startswith is not None and url.startswith(startswith):
                flag = True
            else:
                flag = False
            if endswith is not None and url.endswith(endswith):
                flag = True
            else:
                flag = False

            if flag:
                full_url = link + url
                file_type = url[url.index('.') + 1:]
                save_folder = f'../data/{file_type}_files'
                if not os.path.isdir(save_folder):
                    os.mkdir(save_folder)
                res = requests.get(full_url)
                with open(os.path.join(save_folder, url), 'wb') as f:
                    f.write(res.content)


if __name__ == '__main__':
    speech_url = 'https://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFADVcorpus/Speech/'
    annotations_url = 'https://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFADVcorpus/Annotations/EAF/'
    download_files(speech_url, startswith='DV')
    download_files(annotations_url, endswith='gaze')