from __future__ import print_function
from apiclient import errors
import pickle
import re
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_files_in_folder(service, page_size, folder_id):

    page_token = None

    results = service.files().list( corpora='allDrives', includeItemsFromAllDrives=True, q='\'1F177oBASIbkwv2dSm59sAvGn1YFURIGa\' in parents', supportsAllDrives=True, pageSize=page_size).execute()

    items = results.get('files', [])
    ids = list(map(lambda e: e['id'], items))

    return ids

def get_transcript(service, id):

    # Get the transcript file from the Google Drive API
    content = service.files().get_media(fileId=id).execute().decode('latin-1')

    # Remove timestamps
    transcript = " ".join(re.findall('(?:[0-9]{2}\:[0-9]{2})\r\n(.*)\r', content))

    return transcript

def main():

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    ids = get_files_in_folder(service, 1, '1VJmDJwNZ7mFwcBBPVsioVXtOBs6LMa9X')

    transcript = get_transcript(service, ids[0])

    f = open('training/transcript.txt', 'w')
    f.write(transcript)
    f.close()