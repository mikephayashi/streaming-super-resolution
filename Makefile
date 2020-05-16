m?= update

run:
	python3 download_yt.py -s "programming tutorials" -c ./client.json

git:
	git pull origin master
	git add .
	git commit -am "$(m)"
	git push origin master