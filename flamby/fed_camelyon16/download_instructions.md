## Camelyon16

Camelyon16 as Camelyon17 are open access (CC0), the original dataset is accessible [here](https://camelyon17.grand-challenge.org/Data/).
We will use the nice [pydrive](https://pythonhosted.org/PyDrive) utility to download it easily from Google Drive.


It requires having a gmail account and accessing the console in order to get a json containing an OAuth2.0 secret. 
All steps are described in great details in pydrive's [quickstart](https://pythonhosted.org/PyDrive/quickstart.html),
that we summarize here:
Step 1: get a secret json file by registering an app in Google's console (you can call the app flamby ;) )
Step 2: Download the json file
Step 3: Copy the file in you working directory 
```
cp ~/Downloads/code_secret_client_bignumber.apps.googleusercontent.com.json client_secrets.json
````
Step 4: execute the following lines:
```
from pydrive.auth import GoogleAuth

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
```
This will launch a local browser prompting you for an authorization that you need to give.
Now we will be able to use pydrive to list files in directory and download them.

