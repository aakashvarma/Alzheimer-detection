# 200 -- everything went okay, and the result has been returned (if any)
# 301 -- the server is redirecting you to a different endpoint. This can happen when a company switches domain names, or an endpoint name is changed.
# 401 -- the server thinks you're not authenticated. This happens when you don't send the right credentials to access an API (we'll talk about authentication in a later post).
# 400 -- the server thinks you made a bad request. This can happen when you don't send along the right data, among other things.
# 403 -- the resource you're trying to access is forbidden -- you don't have the right permissions to see it.
# 404 -- the resource you tried to access wasn't found on the server.

import requests
import json
import os
from PIL import Image

url = 'http://127.0.0.1:8000/image/api'

# get data from the API
def getData(link):
    response = requests.get(link)
    data = response.json()
    return data['path']

def extractImage():
    nodePath = getData(url)
    imagePath =  = '/Users/aakashvarma/Documents/Coding/Med-I/' + nodePath
    myimage = Image.open(filename)
    myimage.load()


extractImage()







