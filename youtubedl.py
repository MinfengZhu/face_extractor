from __future__ import unicode_literals
import youtube_dl

class YouTubeDownloader:
    def __init__(self):
        self.ydl_opts = {
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'format': 'bestvideo+bestaudio',
            #'merge-output-format': 'bestvideo+bestaudio[ext=mp4]',
            'outtmpl': 'results/%(id)s.%(ext)s',
            'postprocessors': [
                {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4',}
            ],
            'keepvideo': True,
            #'no_warnings': True,
            }
    def download(self, youtube_link, output_path=None):
        if output_path != None:
            self.ydl_opts['outtmpl'] =  output_path
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            ydl.download([youtube_link])

if __name__ == "__main__":
    yt_dl = YouTubeDownloader()
    yt_dl.download('https://www.youtube.com/watch?v=8ZKzx1C4-DY')
