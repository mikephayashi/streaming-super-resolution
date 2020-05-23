# streaming-super-resolution
Reconstructs streamed low res to high res video

##### [Michael's Colab](https://colab.research.google.com/drive/14aq0YhkuuyEN0FXvNIfEwVEwdBC8nbSg#scrollTo=GytH-0oJXJkX)

- [streaming-super-resolution](#streaming-super-resolution)
        - [Michael's Colab](#michaels-colab)
  - [General notes](#general-notes)
  - [GCP](#gcp)
  - [train.py](#trainpy)
  - [eval.py](#evalpy)
  - [disp.py](#disppy)
  - [download_yt.py:](#downloadytpy)
  - [change_resolution.py:](#changeresolutionpy)

## General notes

* [Conv Transpose Output Dimensions](https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose)
* AE & VQVAE work
  * VAE train DOES NOT

## GCP
* SSH `ssh -i ./ssh_keys/key mikephayashi@34.82.82.235`
  * Alt: `ssh -i ./ssh_keys/key mikephayashi@35.233.175.213`
* Key Generation for gcp instance: `ssh-keygen -t rsa -f ./ssh_keys/key -C mikephayashi` [Ref](https://www.youtube.com/watch?v=2ibBF9YqveY)
* Transferring files: 
  * Local to vm (run on local)`scp -i ssh_keys/key ./client_secrets.json mikephayashi@34.82.82.235:~/streaming-super-resolution/logs/VQVAE`
  * Vm to local (run on local): `scp -r -i ssh_keys/key mikephayashi@35.233.175.213:~/streaming-super-resolution/params/VAE /Users/michaelhayashi/Desktop/GCP`
* `git reset --hard`
* [Gnu screen](https://stackoverflow.com/questions/48221807/google-cloud-instance-terminate-after-close-browser) Detach main processes, can cloe ssh without terminating process

## train.py
* Usage: `python3 train.py -n VAE/VQVAE`

## eval.py
* Usage: `python3 train.py -n VAE/VQVAE`

## disp.py
* Usage: `python3 train.py -n VAE/VQVAE`

## download_yt.py:

Default output: 640x320

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
**Change resolution commented out, uses pytorch change resolution**
- Default
  - Width: 256
  - Height: 256
Usage: `Usage: python3 change_resolution.py [optional]-w <width> [optional]-h <height> -s/--skip= <num to skip default=6>`

Outputs to `res/frames/<name>` (extracted frames of video) and to `res/resized/<name>` (changed resolution of extractedframes)