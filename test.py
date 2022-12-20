import requests

#url = 'http://localhost:8080/predict'
url = 'http://localhost:9696/predict'
# url = 'http://a47d35dc631094cd684e7106f130b3c8-1466140122.ap-southeast-1.elb.amazonaws.com/predict'

data = {'url': 'https://raw.githubusercontent.com/ChanTheDataExplorer/kitchenware-classification-project/main/testing_images/0000.jpg'}

result = requests.post(url, json=data).json()
print(result)