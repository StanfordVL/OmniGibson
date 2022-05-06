from asyncio import subprocess
import subprocess
import sys

THREEDSMAX_PATH = r"C:\Program Files\Autodesk\3ds Max 2022\3dsmaxbatch.exe"

if __name__ == "__main__":
    subprocess.run([THREEDSMAX_PATH] + sys.argv[1:])