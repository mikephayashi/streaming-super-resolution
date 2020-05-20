m?= update

run:
	python3 download_yt.py -n 1 -s "programming tutorials" -c ./client.json

run-all:
	python3 download_yt.py -n 200 -s "programming tutorials" -c ./client.json
	python3 change_resolution.py -n test -v ./res/youtube_vids/tutvideo.mp4
	python3 main.py
	
test-all:
	python3 download_yt.py -n 2 -s "programming tutorials" -c ./client.json
	python3 change_resolution.py -n test -v ./res/youtube_vids/tutvideo.mp4
	python3 main.py

long:
	python3 download_yt.py -n 70 -e standard -u short -s "programming tutorials" -c ./client.json

git:
	git pull origin master
	git add .
	git commit -am "$(m)"
	git push origin master
	echo https://github.com/mikephayashi/streaming-super-resolution

clear-all:
	rm -rf ./res

clear-params:
	rm -rf ./params