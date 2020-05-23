import time

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

start = time.time()

gauth = GoogleAuth()
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)

file1 = drive.CreateFile({'title': 'video.avi'})
file1.SetContentFile("./video.avi")
file1.Upload() # Files.insert()

end = time.time()

print(end-start)