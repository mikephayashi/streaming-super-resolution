m?= update

run:
	python3 download_yt.py -n 1 -s "programming tutorials" -c ./client.json

vids:
	python3 download_yt.py -r 30 -a 10 -s "programming tutorials" -c ./client.json

	
get-test:
	python3 download_yt.py -r 2 -a 1 -s "programming tutorials" -c ./client.json
	python3 change_resolution.py 

long:
	python3 download_yt.py -n 70 -e standard -u short -s "programming tutorials" -c ./client.json

git:
	git pull origin master
	git add .
	git commit -am "$(m)"
	git push origin master
	echo https://github.com/mikephayashi/streaming-super-resolution

clear-res:
	rm -rf ./res

clear-params:
	rm -rf ./params

clear-logs:
	rm -rf ./logs

clear-all:
	rm -rf ./res
	rm -rf ./params
	rm -rf ./logs

pull:
	mv ./res ../
	git reset --hard
	git pull
	mv ../res ./
	