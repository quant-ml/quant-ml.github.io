---
layout: post
title:  "Google Maps API: простой запрос по адресу"
date:   2021-05-23 16:05:36 +0300
categories: jekyll update
---
Задача собрать информацию (в этом примере - координаты) по нескольким физическим адресам через API *Google Maps*. Возьмем адреса нескольких ведущих москвоских ВУЗов, соберем координаты, визуализируем на карте.

<pre><code>
adrDict = {'MSU': '119991, Российская Федерация, Москва, Ленинские горы, д. 1',
           'MIPT': '41701, Московская область, г. Долгопрудный, Институтский переулок, д.9.',
           'HSE': 'Россия, 101000, г. Москва, ул. Мясницкая, д. 20'}
</code></pre>

Выполнить это можно в несколько шагов.

##### **ШАГ 1**: заберем свой API ключ Google Maps.

Сделать это можно на сайте **Google Cloud Platform**. Для этого:

1. Идем на сайт [Google Cloud Credentials](https://console.cloud.google.com/project/_/apiui/credential)

2. Переходим *+ CREATE CREDENTIALS* -> *API key*

3. Далее необходимо авторизовать API для сервиса Geocoding. для этого переходим на [Google Maps Platform](https://console.cloud.google.com/google/maps-apis/overview) -> *APIs* , ищем Geocoding and кликаем *Google Maps Geocoding API* -> *Enable API*

После этого можно сохранить ключ у себяи загрузить его в переменную.

<pre><code>
with open('d/My Drive/googleMap/apikey.txt') as f: # don't forget to update your key location
    apiKey = f.readline()
    f.close
</code></pre>

## **ШАГ 2**: отправить запрос и получить ответ через  **REQUEST**

Одна из самых распростаненных библиотек для отправки / получения запросов [request](https://docs.python-requests.org/en/master/)

<pre><code>
import requests # простая HTTP библиотека для запросов
from collections import defaultdict # удобная фукнция для хранения массивов даже без заранее прописанного ключа
</code></pre>

Теперь пропишем функцию для того, чтобы собрать все входящие в одном удобном датафрейме.

<pre><code>
def getInfo(adrDict):
    '''
    забираем и сохраняем в json координаты с google maps
    '''
    for key in adrDict.keys(): # для каждого указанного места
        query = adrDict[key] # берем физический адрес
        # и вставляем в запрос
        url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + query + '&lang=ru&key=' + apiKey
        data[key] = requests.get(url).json() # сохраняем результат как json

    return data

data = defaultdict(dict) # обработанные координаты кидаем в один словарь
getInfo(adrDict)
</code></pre>

По результату запроса получим следующий выход (в формате json 
<pre><code>
defaultdict(dict,
            {'HSE': {'results': [{'address_components': [{'long_name': '20',
                  'short_name': '20',
                  'types': ['street_number']},
                 {'long_name': 'Myasnitskaya Ulitsa',
                  'short_name': 'Myasnitskaya Ulitsa',
                  'types': ['route']},
                 {'long_name': 'Tsentralnyy administrativnyy okrug',
                  'short_name': 'Tsentralnyy administrativnyy okrug',
                  'types': ['political',
                   'sublocality',
                   'sublocality_level_1']},
                 {'long_name': 'Moskva',
                  'short_name': 'Moskva',
                  'types': ['locality', 'political']},
                 {'long_name': 'Basmannyy',
                  'short_name': 'Basmannyy',
                  'types': ['administrative_area_level_3', 'political']},
                 {'long_name': 'Moskva',
                  'short_name': 'Moskva',
                  'types': ['administrative_area_level_2', 'political']},
                 {'long_name': 'Russia',
                  'short_name': 'RU',
                  'types': ['country', 'political']},
                 {'long_name': '101000',
                  'short_name': '101000',
                  'types': ['postal_code']}],
                'formatted_address': 'Myasnitskaya Ulitsa, 20, Moskva, Russia, 101000',
                'geometry': {'location': {'lat': 55.7615816, 'lng': 37.633323},
                 'location_type': 'ROOFTOP',
                 'viewport': {'northeast': {'lat': 55.7629305802915,
                   'lng': 37.63467198029149},
                  'southwest': {'lat': 55.7602326197085,
                   'lng': 37.6319740197085}}},
                'place_id': 'ChIJhzwS4l1KtUYRapWlISMo4Ek',
                'plus_code': {'compound_code': 'QJ6M+J8 Basmanny District, Moscow, Russia',
                 'global_code': '9G7VQJ6M+J8'},
                'types': ['street_address']}],
              'status': 'OK'},
			--- ... ---
</code></pre>

Теперь визуализируем это на карте

##### **ШАГ 3**: визиализируем с помочщью *Folium*

Для удобства долготу и ширину географических объектов размещаю в датафрейме

<pre><code>
import pandas as pd

adrDf = pd.DataFrame(
    columns = ['Address', 'Latitude', 'Longtitude'],
    index = list(data.keys())
)

# for every address in our dictionary we collect he latitude and longtitute data
for key in data.keys():
    adrDf.loc[key]['Address'] = adrDict[key] 
    adrDf.loc[key]['Latitude'] = data[key]['results'][0]['geometry']['location']['lat']
    adrDf.loc[key]['Longtitude'] = data[key]['results'][0]['geometry']['location']['lng']
</code></pre>

[Folium](http://python-visualization.github.io/folium/) очень удобная и простая для начинающих библиотека, с помощью которой можно визуализировать адреса на *интерактивной карте*. Нет нужды подгружать дополнительные исходники, просто вызываем функцию *folium.Map*. Передаем координаты, ставим исходный масштаб приближения и готово:

<pre><code>
import folium
from folium.plugins import MarkerCluster # функция анимированной кластеризации

# инициализируем карту
m = folium.Map(
    # берем среднее по кооридантам для центровки
    location = adrDf[['Latitude', 'Longtitude']].mean().to_list(), #
    zoom_start=10 # ставим исходное приближение
)

markerCluster = MarkerCluster().add_to(m) # инициализируем кластеризацию

# собираем данные из датафрейма
for idx in adrDf.index:
    location = (adrDf.loc[idx]['Latitude'], adrDf.loc[idx]['Longtitude']) # наши координаты
    folium.Marker(
        location = location,
        popup = adrDf.loc[idx]['Address'], # имя точки
        tooltip = adrDf.loc[idx]['Address'], # текст поверх точек
    ).add_to(markerCluster)

m # рисуем карту
</code></pre>

Сохраняем в HTML:

<pre><code>
m.save('2021-05-23-google-maps-api-simple-address-querry.html')
</code></pre>

## в итоге получаем карту с адресами:

<img src="{{site.baseurl}}/assets/img/2021-05-23-google-maps-api-simple-address-querry.png">
