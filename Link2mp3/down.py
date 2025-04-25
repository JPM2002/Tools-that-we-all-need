import yt_dlp
import re
import os

FFMPEG_PATH = r"C:\Users\Javie\Desktop\ffmpeg-7.1-full_build\bin"

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def load_urls(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w').close()
        print(f"📄 Created empty input file: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_downloaded(file_path):
    if not os.path.exists(file_path):
        open(file_path, 'w').close()
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def mark_as_downloaded(file_path, url):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def download_as_mp3(url):
    ydl_opts = {
        'ffmpeg_location': FFMPEG_PATH,
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'noplaylist': True,
        'quiet': False,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            },
            {
                'key': 'FFmpegMetadata',
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = sanitize_filename(info.get("title", "output"))
        print(f"✅ Saved as: {title}.mp3")

def batch_download(input_txt='urls.txt', downloaded_txt='downloaded.txt'):
    urls = load_urls(input_txt)
    downloaded = load_downloaded(downloaded_txt)

    for url in urls:
        if url in downloaded:
            print(f"⏩ Skipping (already downloaded): {url}")
            continue

        try:
            download_as_mp3(url)
            mark_as_downloaded(downloaded_txt, url)
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")

# Run the batch
batch_download()
