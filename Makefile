m?= update

run:
	# python3 download_yt.py -s "programming tutorials" -c ./client.json
	# python3 change_resolution.py -n test -v ./res/youtube_vids/tutvideo.mp4
	python3 load_data.py -f test -n 10 -w 100 -h 100
	

long:
	python3 download_yt.py -n 70 -e standard -u short -s "programming tutorials" -c ./client.json

git:
	git pull origin master
	git add .
	git commit -am "$(m)"
	git push origin master
	echo https://github.com/mikephayashi/streaming-super-resolution

clear:
	rm -rf ./res/frames/$(dir)
	rm -rf ./res/resized/$(dir)
