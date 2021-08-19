from __future__ import unicode_literals
import youtube_dl

class YouTubeDownloader:
    def __init__(self):
        self.ydl_opts = {
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'format': 'bestvideo+bestaudio',
            'keepvideo': True,
            'outtmpl': 'results/%(id)s.%(ext)s',
            #'merge-output-format': 'bestvideo+bestaudio[ext=mp4]',
            #'postprocessors': [
            #    {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4',}
            #],
            #'no_warnings': True,
            }
    def download(self, url, output_path=None):
        if output_path != None:
            self.ydl_opts['outtmpl'] =  output_path
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            result = ydl.extract_info("{}".format(url), download=True)
            filename = ydl.prepare_filename(result)
            #filename = ''.join(filename.split(".")[:-1]) + '.mp4'
            return filename
            #ydl.download([url])

if __name__ == "__main__":
    yt_dl = YouTubeDownloader()
    yt_dl.download('https://www.youtube.com/watch?v=8ZKzx1C4-DY')
