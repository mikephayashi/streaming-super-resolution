"""
download_yt.py
Author: Michael Hayashi

Downloads Youtube videos based on fetched keyword
"""

import os
import json
import argparse
import sys
import getopt
import threading
from threading import Semaphore
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from pytube import YouTube  # pip3 (or pip) install pytube3 not pytube

# Thread control
sem = Semaphore(3)  # Limit at 3


def download_video(id):
    """
    id: video id in Youtube
    """
    sem.acquire()
    print("Downloading: ", id)
    yt = YouTube('https://www.youtube.com/watch?v=' + id)
    stream = yt.streams.first()
    stream.download('./youtube_vids')
    sem.release()


def collect_ids(num_vids, json):
    """
    num_vids: number of videos returned in response
    json: video information returned in response
    """
    ids = []
    for i in range(0, num_vids):
        vid_id = json['items'][i]['id']['videoId']
        ids.append(vid_id)
    return ids


def request_vids(page_token, client, search, definition, duration):
    """
    Uses Youtube's API to fetch video info
    Return: (Number of videos fetched, video info as json)
    """

    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = client

    # Get credentials and create an API client
    scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    # First page
    if page_token is None:

        request = youtube.search().list(
            part="snippet",
            maxResults=5,  # 1-50
            order="viewCount",
            q=search,
            type="video",
            videoDefinition=definition,
            videoDuration=duration
        )

    else:
        request = youtube.search().list(
            part="snippet",
            maxResults=50,  # 1-50
            order="viewCount",
            pageToken=str(PAGE_TOKEN),
            q=search,
            type="video",
            videoDefinition=definition,
            videoDuration=duration
        )

    json = request.execute()
    num_vids = json['pageInfo']['resultsPerPage']

    return num_vids, json


def print_args():
    """
    Print cli arguments interface
    """
    print(
        "download_yt.py -s/--search= \"<search term>\" -c/--client <client path> [optional]-e/--definition= <any, high, standard> [optional]-u/--duration= <any,long,medium,short>")
    print("Refer to repo's README.md for more info")


if __name__ == "__main__":

    argv = sys.argv[1:]

    # Possible values
    definitions = ["any", "high", "stadard"]
    durations = ["any", "long", "medium", "short"]

    # Default values
    page_token = None  # Retured in response json
    client = None  # Client.json path (Auth2.0 token in Youtubev3 API)
    search = None  # Search term used to look up videos
    definition = definitions[0]
    duration = durations[0]

    # Parse arguments
    try:
        opts, args = getopt.getopt(
            argv, "hs:e:u:c:", ["search=", "definition=", "duration=", "client="])
    except getopt.GetoptError:
        print_args()

    for opt, arg in opts:
        if opt == '-h':
            print_args()
            sys.exit()
        elif opt in ("-s", "--search"):
            search = arg
        elif opt in ("-e", "--definition"):
            if arg not in definition:
                print("Not valid video definition")
                print_args()
            definition = arg
        elif opt in ("-u", "--duration"):
            if arg not in duration:
                print("Not valid video duration")
                print_args()
            duration = arg
        elif opt in ("-c", "--client"):
            client = arg

    # Exit if no client is specified
    if client is None:
        print("You must specify client")
        print_args()
        sys.exit()

    # Exit if no search term is specififed
    if search is None:
        print("You must specify search term")
        print_args()
        sys.exit()

    num_vids, json = request_vids(page_token, client, search, definition, duration)
    ids = collect_ids(num_vids, json)

    # Spawn about 3 threads to speed up downloads
    threads = []
    for id in ids:
        thread = threading.Thread(target=download_video, args=(id,))
        threads.append(thread)
        thread.start()

    for index, thread in enumerate(threads):
        thread.join()
