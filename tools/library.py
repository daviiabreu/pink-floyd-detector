import argparse
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

from tools.common import ROOT, SAMPLE_RATE, file_hash

STUDIO = {
    "The Piper at the Gates of Dawn": 11,
    "A Saucerful of Secrets": 7,
    "More": 13,
    "Ummagumma": 16,
    "Atom Heart Mother": 5,
    "Meddle": 6,
    "Obscured by Clouds": 10,
    "The Dark Side of the Moon": 10,
    "Wish You Were Here": 5,
    "Animals": 5,
    "The Wall": 26,
    "The Final Cut": 13,
    "A Momentary Lapse of Reason": 11,
    "The Division Bell": 11,
    "The Endless River": 18,
}
AUDIO_SUFFIXES = {".m4a", ".mp3", ".flac", ".wav", ".ogg"}


def normalize(text):
    return unicodedata.normalize("NFC", text).casefold().strip()


def slug(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (
        text
        if len(text) <= 64
        else text[:53] + "_" + hashlib.sha256(text.encode()).hexdigest()[:10]
    )


def executable(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Instale {name} e deixe o executável no PATH")
    return found


def decode(path):
    result = subprocess.run(
        [
            executable("ffmpeg"),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype="<i2").copy()


def scan(root):
    root = Path(root).resolve()
    albums, tracks = {}, []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        relative = path.relative_to(root)
        album = unicodedata.normalize("NFC", relative.parts[0]).removeprefix("Álbum - ")
        result = subprocess.run(
            [
                executable("ffprobe"),
                "-v",
                "error",
                "-show_entries",
                "format=duration:format_tags=artist,title,album:stream=codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata = json.loads(result.stdout)
        info = metadata["format"]
        record = {
            "path": relative.as_posix(),
            "album": album,
            "title": path.stem,
            "duration_s": float(info["duration"]),
            "sha256": file_hash(path),
            "tags": info.get("tags", {}),
            "streams": metadata["streams"],
        }
        tracks.append(record)
        albums[album] = albums.get(album, 0) + 1
    if not tracks:
        raise ValueError("Nenhum arquivo de áudio encontrado")
    conflicts = [
        {"path": track["path"], "artist_tag": track["tags"]["artist"]}
        for track in tracks
        if track["tags"].get("artist")
        and normalize(track["tags"]["artist"]) not in {"pink floyd", "ピンク・フロイド"}
    ]
    return {
        "source": "https://www.pinkfloyd.com/music/",
        "tracks": tracks,
        "track_count": len(tracks),
        "albums": albums,
        "artist_identity_verified": False,
        "artist_tag_conflicts": conflicts,
        "missing_artist_tag_count": sum(not t["tags"].get("artist") for t in tracks),
        "missing_studio_albums": [
            name for name in STUDIO if normalize(name) not in {normalize(album) for album in albums}
        ],
        "expected_track_counts": STUDIO,
        "note": "Inventário por nomes e contagem, com verificação de decodificação na preparação. "
        "The Final Cut inclui When the Tigers Broke Free; Ummagumma conta suas partes separadas. "
        "Nomes e tags não autenticam a edição ou a identidade acústica da gravação.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confere os áudios e atualiza o inventário")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/catalogo-local.json")
    args = parser.parse_args()
    report = scan(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tracks": report["track_count"],
                "missing_studio_albums": report["missing_studio_albums"],
            }
        )
    )
