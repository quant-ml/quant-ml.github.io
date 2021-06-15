---
layout: post
title:  "Google Maps API: simple address query"
date:   2021-05-23 16:05:36 +0300
categories: jekyll update
---
Let's find out some data on some addresses (top tech uninversities) in Moscow with google API and visualize it with the simple map:
<pre><code>
adrDict = {'MSU': '119991, Российская Федерация, Москва, Ленинские горы, д. 1',
           'MIPT': '41701, Московская область, г. Долгопрудный, Институтский переулок, д.9.',
           'HSE': 'Россия, 101000, г. Москва, ул. Мясницкая, д. 20'}
</code></pre>

That will require several simple steps.

### **STEP 1**: we need to get the Google Maps API key.

You can get the Google Maps API key via **Google Cloud Platform**:

1. Go to the [Google Cloud Credentials Page](https://console.cloud.google.com/project/_/apiui/credential)

2. Click *+ CREATE CREDENTIALS* -> *API key*

3. The next step is to enable your API key for Geocoding. To do that navigate to [Google Maps Platform](https://console.cloud.google.com/google/maps-apis/overview) -> *APIs* , search for Geocoding and click on *Google Maps Geocoding API* -> *Enable API*

Finally, save and load your API key:

<pre><code>
with open('d/My Drive/googleMap/apikey.txt') as f: # don't forget to update your key location
    apiKey = f.readline()
    f.close
</code></pre>

### **STEP 2**: send and get **REQUEST**

The most common tool to send HTTP request is [request](https://docs.python-requests.org/en/master/)

<pre><code>
import requests # is an elegant and simple HTTP library for Python, built for human beings (c)
from collections import defaultdict # useful to to make container to create automatic keys if missing
</code></pre>

Now let's have a function to collect that data in one place:

<pre><code>
def getInfo(adrDict):
    '''
    collect coordinates data from google maps
    '''
    for key in adrDict.keys(): # for every address in the dictionary
        query = adrDict[key] # get the respective address from the dictionary
        # make the url for the request
        url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + query + '&lang=en&key=' + apiKey
        data[key] = requests.get(url).json() # collect the data and dump it as json

    return data

data = defaultdict(dict) # place all the data to one single variable
getInfo(adrDict)
</code></pre>

The output will be like this:

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

Now we have the data to visualize.

### **STEP 3**: visualize with *Folium*

To make it more convenient let's store the latitude and longtitute as the dataframe.

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

[Folium](http://python-visualization.github.io/folium/) is the library that helps to visualize data on an interactive leaflet map. It is quite easy to display the address coordinates and no need to upload the map meta data, since you already have it while calling *folium.Map* function.

So we simply initiate the map by passing the mean of our coordinates and the initial zoom of the map to the *folium.Map* function

<pre><code>
import folium
from folium.plugins import MarkerCluster # animated marker Clustering functionality

# init the map
m = folium.Map(
    # let's take the mean of the coordinates to form the map
    location = adrDf[['Latitude', 'Longtitude']].mean().to_list(),  
    zoom_start=3 # initial zoom level for the map
)

markerCluster = MarkerCluster().add_to(m) # init clustering

# collect the data 
for idx in adrDf.index:
    location = (adrDf.loc[idx]['Latitude'], adrDf.loc[idx]['Longtitude']) # coordinates of a map point
    folium.Marker(
        location = location,
        popup = adrDf.loc[idx]['Address'], # label of a map point
        tooltip = adrDf.loc[idx]['Address'], # display a text over the object
    ).add_to(markerCluster)

m # display the map
</code></pre>

Let's save the resulting map into HTML file:
<pre><code>
m.save('2021-05-23-google-maps-api-simple-address-querry.html')
</code></pre>

#### the map with the label markers:
<img src="{{site.baseurl}}/assets/img/2021-05-23-google-maps-api-simple-address-querry.png">
