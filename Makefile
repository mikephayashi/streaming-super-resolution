m?= update

all:
	python3 download_yt.py -r 30 -a 10 -s "domics" -c ./client.json
	python3 change_resolution.py
	python3 train.py

vids:
	python3 download_yt.py -r 30 -a 10 -s "programming tutorials" -c ./client.json

login:
	ssh -i ./ssh_keys/key mikephayashi@35.233.175.213

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

clear-all:
	rm -rf ./params
	rm -rf ./logs

clear-params:
	rm -rf ./params

clear-logs:
	rm -rf ./logs
	
transfer:
	scp -r -i ssh_keys/key mikephayashi@35.233.175.213:~/streaming-super-resolution/logs/VQVAE /Users/michaelhayashi/Desktop/GCP
	scp -r -i ssh_keys/key mikephayashi@35.233.175.213:~/streaming-super-resolution/params/VQVAE /Users/michaelhayashi/Desktop/GCP
	scp -r -i ssh_keys/key mikephayashi@35.233.175.213:~/streaming-super-resolution/reconstructed/VQVAE /Users/michaelhayashi/Desktop/GCP

background:
	screen -r

terminate:
	# screen -X -S [session # you want to kill] quit