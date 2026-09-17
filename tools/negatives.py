import argparse
import hashlib
import json
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.common import ROOT


def split_overlap(groups):
    assignments = {}
    for split, sources in groups.items():
        for source in set(sources):
            assignments.setdefault(source, []).append(split)
    return {source: splits for source, splits in assignments.items() if len(splits) > 1}


def manifest_overlap(tracks):
    names, groups = {}, {}
    for track in tracks:
        name = names.setdefault(track["sha256"], track["path"])
        groups.setdefault(track["split"], []).append(name)
    return split_overlap(groups)


def download(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "docs/negative-music.json").read_text())
    tracks = {}
    for track in manifest["tracks"]:
        previous = tracks.setdefault(track["path"], track)
        if previous["sha256"] != track["sha256"]:
            raise ValueError(f"Mesmo destino com arquivos diferentes: {track['path']}")
    overlap = manifest_overlap(manifest["tracks"])
    if overlap:
        warnings.warn(f"Negativos compartilhados entre divisões: {overlap}", stacklevel=2)

    def fetch(track):
        target = output / track["path"]
        if not target.exists():
            content = urllib.request.urlopen(track["url"], timeout=60).read()
            if hashlib.sha256(content).hexdigest() != track["sha256"]:
                raise ValueError(f"Arquivo remoto mudou: {track['title']}")
            target.write_bytes(content)
        elif hashlib.sha256(target.read_bytes()).hexdigest() != track["sha256"]:
            raise ValueError(f"Arquivo local mudou: {target}")
        return target

    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, _ in enumerate(pool.map(fetch, tracks.values()), start=1):
            if index % 10 == 0:
                print(f"Negativos: {index}/{len(tracks)}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baixa as músicas negativas CC BY 4.0 do manifesto"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/negative_music")
    download(parser.parse_args().output)
