import yt_dlp
ydl_opts = {'quiet': True, 'skip_download': True, 'js_runtimes': {'node': {}}}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info('https://www.youtube.com/watch?v=ZXQFKoC5UgY', download=False)
    for f in info.get('formats', []):
        fid = f.get('format_id', '')
        ext = f.get('ext', '')
        h = f.get('height', '')
        vc = f.get('vcodec', '')
        ac = f.get('acodec', '')
        fs = f.get('filesize', 0)
        print(f'{fid:>10} {ext:>5} {str(h):>4} {vc:>10} {ac:>10} {fs}')