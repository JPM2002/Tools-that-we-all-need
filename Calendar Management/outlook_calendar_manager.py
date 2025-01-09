from msal import ConfidentialClientApplication
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

# Replace with your Azure app credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")

# Verify variables (optional)
if not all([CLIENT_ID, CLIENT_SECRET, TENANT_ID]):
    raise ValueError("Missing one or more Azure app credentials in the .env file.")

def get_access_token():
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=authority,
        client_credential=CLIENT_SECRET,
    )
#Need to add more scopes to access more data to the app.
    scopes = ["https://graph.microsoft.com/.default"]
    token_response = app.acquire_token_for_client(scopes=scopes)
    if "access_token" in token_response:
        return token_response["access_token"]
    else:
        raise Exception("Could not acquire token")

import requests

def list_events():
    access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    url = "https://graph.microsoft.com/v1.0/me/events"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        events = response.json().get('value', [])
        for event in events:
            print(f"Subject: {event['subject']}, Start: {event['start']['dateTime']}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

import datetime

def create_event():
    access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    event = {
        "subject": "Team Meeting",
        "start": {
            "dateTime": "2024-12-28T10:00:00",
            "timeZone": "UTC"
        },
        "end": {
            "dateTime": "2024-12-28T11:00:00",
            "timeZone": "UTC"
        },
        "body": {
            "contentType": "HTML",
            "content": "Team meeting to discuss project updates."
        },
        "attendees": [
            {
                "emailAddress": {
                    "address": "teammate@example.com",
                    "name": "Teammate"
                },
                "type": "required"
            }
        ]
    }

    url = "https://graph.microsoft.com/v1.0/me/events"
    response = requests.post(url, headers=headers, json=event)
    if response.status_code == 201:
        print("Event created successfully!")
    else:
        print(f"Error: {response.status_code}, {response.text}")



if __name__ == "__main__":
    try:
        # Step 1: Test Access Token Retrieval
        print("Testing access token retrieval...")
        token = get_access_token()
        print("Access token retrieved successfully!")

        # Step 2: List Existing Events
        print("\nListing existing events...")
        list_events()

        # Step 3: Create a New Event
        print("\nTesting event creation...")
        create_event()
    except Exception as e:
        print(f"An error occurred: {e}")

