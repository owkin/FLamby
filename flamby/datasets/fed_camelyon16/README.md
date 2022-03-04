## Camelyon16

Camelyon16 as Camelyon17 are open access (CC0), the original dataset is accessible [here](https://camelyon17.grand-challenge.org/Data/).
We will use the [Google-Drive-API-v3](https://developers.google.com/drive/api/v3/quickstart/python) in order to fetch
the slides from the public Google Drive and will then tile the matter using a feature extractor producing a bag of features for each slide.


In order to use the Google Drive API we need to have a gmail account and to access the [google developpers console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) in order to get a json containing an OAuth2.0 secret.  

All steps necessary to obtain the JSON are described in numerous places in the internet such as in pydrive's [quickstart](https://pythonhosted.org/PyDrive/quickstart.html),
or in this [very nice tutorial's first 5 minutes](https://www.youtube.com/watch?v=1y0-IfRW114) on Youtube.
It takes 5 minutes and the important steps are listed below:

Step 1: Create a project in [Google console](https://console.cloud.google.com/apis/credentials/consent?authuser=1) you can call it flamby for instance   
Step 2: Go to Oauth2 consent screen (on the left of the webpage), choose a name for your app and publish it for external use   
Step 3: Go to Credentials, create id, then client oauth id  
Step 4: Choose Web app, go through the steps and **allow URI redirect** towards http://localhost:6006 and http://localhost:6006/ (notice the last backslash)  
Step 5: retrieve the secrets in JSON by clicking on Download icon at the end of the process  
Then copy-paste your secrets to the directory you want:
```
cp ~/Downloads/code_secret_client_bignumber.apps.googleusercontent.com.json client_secrets.json
````
**If you have ssh to a distant server make sure you do ssh forwarding of the port 6006 onto the port 6006 of your laptop.**
Make sure you have enough space to hold the dataset (900G).
Then run:
```
python download.py --output-folder ./camelyon16_slides --path-to-secret /path/towards/client_secrets.json --port 6006
```
This will download all of Camelyon's slides in `./camelyon16_slides.`


The next step is to tile the matter on each slide with a feature extractor pretrained on IMAGENET.
We will use histolab to segment the matter on each slide and torchvision to download a pretrained ResNet50.




