import argparse
import json
import re
import struct
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import serial

from tools.common import FRAME_SIZE, SAMPLE_RATE, file_hash

PACKET_BYTES = 8 + FRAME_SIZE * 2


def serial_open(port, baudrate):
    device = serial.Serial(port=None, baudrate=baudrate, timeout=1)
    device.dtr = False
    device.rts = False
    device.port = port
    device.open()
    return device


def parse_packet(buffer):
    offset = buffer.find(b"AUD0")
    if offset < 0:
        del buffer[:-3]
        return None
    if offset:
        del buffer[:offset]
    if len(buffer) < PACKET_BYTES:
        return None
    sequence = struct.unpack_from("<I", buffer, 4)[0]
    pcm = bytes(buffer[8:PACKET_BYTES])
    del buffer[:PACKET_BYTES]
    return sequence, pcm


def destination(args):
    for name in [args.label, args.session]:
        if not re.fullmatch(r"[a-z0-9_]{1,64}", name):
            raise ValueError("Rótulo e sessão: letras minúsculas sem acentos, números e underscore")
    path = args.data / args.split / args.label / f"{args.session}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    return path


def record(args):
    if not 3 <= args.seconds <= 300:
        raise ValueError("Use entre 3 e 300 segundos por sessão")
    path = destination(args)
    frames, buffer, previous = [], bytearray(), None
    needed = int(np.ceil(args.seconds * SAMPLE_RATE / FRAME_SIZE))
    deadline = time.monotonic() + args.seconds + 20
    print(f"Gravando {args.label} por {args.seconds}s. Reproduza o trecho agora.")
    with serial_open(args.port, 921600) as device:
        while len(frames) < needed:
            if time.monotonic() > deadline:
                raise TimeoutError("Sem áudio suficiente; confira porta e firmware 'record'")
            buffer.extend(device.read(device.in_waiting or 1))
            while (packet := parse_packet(buffer)) is not None:
                sequence, pcm = packet
                if previous is not None and (sequence - previous) % (2**32) != 1:
                    raise ValueError(
                        "Perda de pacotes de áudio. A sessão não foi salva; grave novamente."
                    )
                previous = sequence
                frames.append(pcm)
                if len(frames) >= needed:
                    break
    pcm_bytes = b"".join(frames)
    pcm = np.frombuffer(pcm_bytes, dtype="<i2").astype(float) / 32768
    rms = float(np.sqrt(np.mean((pcm - pcm.mean()) ** 2)))
    clipping = float(np.mean(np.abs(pcm) > 0.98))
    if rms < 1e-5:
        raise ValueError("Áudio praticamente zerado: confira canal L/R, fios e alimentação")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        stream.writeframes(pcm_bytes)
    info = {
        "source": "INMP441",
        "rms": rms,
        "clipping_fraction": clipping,
        "samples": len(pcm),
        "sha256": file_hash(path),
        "split": args.split,
        "session": args.session,
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Salvo: {path} | RMS={rms:.6f} | clipping={clipping:.2%}")


def import_audio(args):
    if not args.source.is_file():
        raise ValueError("Informe um arquivo de áudio local")
    if args.start < 0 or not 3 <= args.seconds <= 300:
        raise ValueError("Início deve ser positivo e a duração entre 3 e 300 segundos")
    args.split = "train"
    path = destination(args)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-ss",
            str(args.start),
            "-i",
            str(args.source),
            "-t",
            str(args.seconds),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "local_file",
                "original_sha256": file_hash(args.source),
                "start_seconds": args.start,
                "requested_seconds": args.seconds,
                "split": "train",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Importado para treino: {path}. Use gravações independentes para val/test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coleta no INMP441 ou importa áudio local para treino"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["record", "import"]:
        sub = commands.add_parser(name)
        sub.add_argument("--data", type=Path, default=Path("data/pink_floyd"))
        sub.add_argument("--label", required=True)
        sub.add_argument("--session", required=True)
        sub.add_argument("--seconds", type=float, default=30)
        if name == "record":
            sub.add_argument("--port", required=True)
            sub.add_argument("--split", choices=["train", "val", "test"], required=True)
        else:
            sub.add_argument("--source", type=Path, required=True)
            sub.add_argument("--start", type=float, default=0)
    args = parser.parse_args()
    record(args) if args.command == "record" else import_audio(args)
