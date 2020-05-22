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
log_sem = Semaphore(1)


def download_video(id, data_type):
    """
    id: video id in Youtube
    """
    sem.acquire()
    print("Downloading: ", id)
    log_sem.acquire()
    with open("logs/videos.txt", "a") as file:
        file.write("Train: {id}\n".format(id=id))
        file.close()
    log_sem.release()
    yt = YouTube('https://www.youtube.com/watch?v=' + id)
    stream = yt.streams.first()
    if data_type == "train":
        stream.download('./res/youtube_vids/train')
    elif data_type == "test":
        stream.download('./res/youtube_vids/test')
    sem.release()


def collect_video_info(num_vids, json):
    """
    num_vids: number of videos returned in response
    json: video information returned in response
    return ids: video ids
    return page_token: id which page
    """
    next_page_token = json['nextPageToken']
    ids = []
    for i in range(0, num_vids):
        vid_id = json['items'][i]['id']['videoId']
        ids.append(vid_id)
    return ids, next_page_token


def request_vids(requested_num, page_token, client, search, definition, duration):
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
            maxResults=requested_num,  # 1-50
            order="viewCount",
            q=search,
            type="video",
            videoDefinition=definition,
            videoDuration=duration
        )

    else:
        request = youtube.search().list(
            part="snippet",
            maxResults=requested_num,  # 1-50
            order="viewCount",
            pageToken=page_token,
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
        "download_yt.py -n/--number= <number of vidoes> -s/--search= \"<search term>\" -c/--client <client path> [optional]-e/--definition= <any, high, standard> [optional]-u/--duration= <any,long,medium,short>")
    print("Refer to repo's README.md for more info")


if __name__ == "__main__":

    argv = sys.argv[1:]

    # Possible values
    definitions = ["high", "any", "standard"]
    durations = ["medium", "long", "short", "any"]

    # Default values
    next_page_token = None  # Retured in response json
    client = None  # Client.json path (Auth2.0 token in Youtubev3 API)
    search = None  # Search term used to look up videos
    definition = definitions[0]
    duration = durations[0]
    num_train = 30  # Number of videos to download
    num_test = 10

    if not os.path.exists("./logs"):
        os.makedirs("./logs")


    # Parse arguments
    try:
        opts, args = getopt.getopt(
            argv, "hr:a:s:e:u:c:", ["train=", "test=", "search=", "definition=", "duration=", "client="])
    except getopt.GetoptError:
        print_args()

    for opt, arg in opts:
        if opt == '-h':
            print_args()
            sys.exit()
        elif opt in ("-n", "--number"):
            num_left = int(arg)
        elif opt in ("-r", "--train"):
            num_train = int(arg)
        elif opt in ("-a", "--test"):
            num_test = int(arg)
        elif opt in ("-s", "--search"):
            search = arg
        elif opt in ("-e", "--definition"):
            if arg not in definitions:
                print("Not valid video definition")
                print_args()
                sys.exit()
            definition = arg
        elif opt in ("-u", "--duration"):
            if arg not in durations:
                print("Not valid video duration")
                print_args()
                sys.exit()
            duration = arg
        elif opt in ("-c", "--client"):
            client = arg

    if not os.path.exists("./res/youtube_vids/train"):
        os.makedirs("./res/youtube_vids/train")
    if not os.path.exists("./res/youtube_vids/test"):
        os.makedirs("./res/youtube_vids/test")

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

    num_left = num_train + num_test

    while num_left > 0:

        # Max request is at 50, so keep requesting until 0
        requested_num = 0
        if num_left > 50:
            requested_num = 50
            num_left -= 50
        else:
            requested_num = num_left
            num_left = 0

        num_vids, json = request_vids(
            requested_num, next_page_token, client, search, definition, duration)
        ids, next_page_token = collect_video_info(num_vids, json)

        # Spawn 3 threads to speed up downloads
        threads = []
        for id in ids:
            if num_train > 0:
                thread = threading.Thread(target=download_video, args=(id,"train",))
                num_train -= 1
            elif num_test > 0:
                thread = threading.Thread(target=download_video, args=(id,"test",))
                num_test -= 1
            threads.append(thread)
            thread.start()

        for index, thread in enumerate(threads):
            thread.join()
