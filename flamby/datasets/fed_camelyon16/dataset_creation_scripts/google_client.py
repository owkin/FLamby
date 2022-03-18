import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def create_service(
    client_secret_file="./client_secrets.json",
    api_name="drive",
    api_version="v3",
    scopes=["https://www.googleapis.com/auth/drive"],
    port=6006,
):
    """Instantiate a client that is able to upload and download data from \
        Google Drives using downloaded secrets from the Google console.
    Inspired from: https://developers.google.com/drive/api/v3/quickstart/python

    Parameters
    ----------
    client_secret_file : str, optional
        The json that can be downloaded after completing OAuth2 for a \
        published app, by default "./client_secrets.json"
    api_name : str, optional
        The name of the API to use, by default "drive"
    api_version : str, optional
        The version of the API to use, by default "v3"
    scopes : list, optional
        The permissions scope of the client, by default \
        ["https://www.googleapis.com/auth/drive"]
    port : int, optional
        The port for URI redirect, by default 8080

    Returns
    -------
    drive
        The client to use to make queries. Also stores a pickled version of \
        the secret for easy access.
    """
    print(client_secret_file, api_name, api_version, scopes, sep="-")
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = scopes
    print(SCOPES)

    cred = None

    pickle_file = f"token_{API_SERVICE_NAME}_{API_VERSION}.pickle"

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server(port=port)

        with open(pickle_file, "wb") as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, "service created successfully")
        return service
    except Exception as e:
        print("Unable to connect.")
        print(e)
        return None
