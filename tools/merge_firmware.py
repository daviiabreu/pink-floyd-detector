import subprocess
from pathlib import Path

from SCons.Script import DefaultEnvironment


def merge_firmware(source, target, env):
    build = Path(env.subst("$BUILD_DIR"))
    esptool = Path(env.PioPlatform().get_package_dir("tool-esptoolpy")) / "esptool.py"
    subprocess.run(
        [
            env.subst("$PYTHONEXE"),
            str(esptool),
            "--chip",
            "esp32",
            "merge_bin",
            "-o",
            str(build / "wokwi.bin"),
            "--flash_mode",
            "dio",
            "--flash_freq",
            "40m",
            "--flash_size",
            "4MB",
            "0x1000",
            str(build / "bootloader.bin"),
            "0x8000",
            str(build / "partitions.bin"),
            "0x10000",
            str(build / "firmware.bin"),
        ],
        check=True,
    )


env = DefaultEnvironment()
env.AddCustomTarget(
    name="wokwi",
    dependencies=["$BUILD_DIR/${PROGNAME}.bin"],
    actions=[merge_firmware],
    title="Build Wokwi image",
    description="Merge bootloader, partition table and application",
)
