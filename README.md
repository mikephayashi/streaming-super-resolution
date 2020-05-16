# streaming-super-resolution
Reconstructs streamed low res to high res video


- [streaming-super-resolution](#streaming-super-resolution)
  - [download_yt.py:](#downloadytpy)
  - [change_resolution.py:](#changeresolutionpy)
  - [main.py:](#mainpy)


## download_yt.py:

For default run:
- `make run` 
- Client = ./client.json
- Search = "programming tutorials"

For other runs:
`download_yt.py -n/--number= <number of vidoes> -s/--search= \"<search term>\" -c/--client <client path> [optional]-e/--definition= <any, high, standard> [optional]-u/--duration= <any,long,medium,short>`
- Shortform ex. `python3 download_yt.py -s "programming tutorials" -c ./client.json`
- Longform equivalent. `python3 download_yt.py --search="programming tutorials" -client=./client.json`
- **Must wrap `<search term>` in quotation marks to capture multiple keywords**

* Youtube API client
  
1. Go to [API library](https://console.developers.google.com/apis/library?project=fluted-equinox-277319&folder&organizationId)
2. Search for *YouTube Data API v3*
3. In the left hand bar, go to *credentials*
4. Near the top, click *create credentials*
5. Click *OAuth client ID*
   1. this may prompt you the 'OAuth Consent Screen', make it `external`, consent, then make the Oauth client ID
6. Choose *Desktop app* for application type
7. Then, create
8. Go to credentials
9. Downlaod your newly made Oauth client ID
10. Drag it into the root folder of this project and renmae it to `client.json`
11. Now, you can run download_yt.py

* Each Youtube API call incurs a quota cost of 100. Free quotas are limited to 10,000 daily. Each time this file is ran, it will make a request for every 50 videls. 

"Visiting the url to authorize this application" may tell you "This app isn't verified". If so, click advance (may be different if not in Chrome) and click "Go to `<app name>`(unsafe)". This is fine since this is our app. Allow permissions.

## change_resolution.py:
Use later on to change video resolution

## main.py:
Empty