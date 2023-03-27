
from __future__ import print_function
import os.path
from typing import List
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pprint import pprint


# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# The ID and range of a sample spreadsheet.
PROJECT_FOLDER_ID = "1lmXX_VbuVqfAaZXLa14uTfJN8eirCRQ2"
TEMPLATE_FILE_ID = "1WYUMrrn6dpkyyzhWSK_MXf7mpU4qHrPvbNWocQZiJjo"


def colnum_string(n):
    string = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


def string_range(start_row, start_column, end_row, end_column):
    return colnum_string(start_column) + str(start_row) + \
        ':' + colnum_string(end_column) + str(end_row)


def export_to_google_sheets(
    records: List[List[float]] = None,
    labels: List[str] = None,
    title: str = "data",
    folder_id: str = PROJECT_FOLDER_ID,
    template_id: str = TEMPLATE_FILE_ID,
):
    if labels:
        records = [labels] + records

    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service_drive = build('drive', 'v3', credentials=creds)
        service_sheets = build('sheets', 'v4', credentials=creds)

        # Call the Drive v3 API
        request = service_drive \
            .files() \
            .copy(fileId=template_id,
                  body={
                      "parents": [
                          {
                              "kind": "drive#fileLink",
                              "id": folder_id,
                          }
                      ],
                      'name': title,
                  }
                  )
        result = request.execute()
        spreadsheet_id = result["id"]

        sheet = service_sheets.spreadsheets()
        # The A1 notation of a range to search for a logical table of data.
        # Values will be appended after the last row of the table.
        # TODO: Update placeholder value.
        range_ = string_range(1, 1, len(records), len(records[0]))

        # How the input data should be inserted.
        insert_data_option = 'OVERWRITE'  # TODO: Update placeholder value.

        # How the input data should be interpreted.
        # TODO: Update placeholder value.
        value_input_option = 'RAW'

        value_range_body = {
            'values': records
        }

        request = sheet \
            .values() \
            .append(
                spreadsheetId=spreadsheet_id,
                range=range_,
                valueInputOption=value_input_option,
                insertDataOption=insert_data_option,
                body=value_range_body
            )
        response = request.execute()
        print(
            f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}')

    except HttpError as err:
        print(err)
