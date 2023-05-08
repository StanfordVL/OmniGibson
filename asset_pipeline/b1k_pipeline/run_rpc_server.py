import subprocess
import b1k_pipeline.utils

THREEDSMAX_PATH = r"C:\Program Files\Autodesk\3ds Max 2022\3dsmaxbatch.exe"
RPC_SERVER_PATH = b1k_pipeline.utils.PIPELINE_ROOT / "b1k_pipeline/max/rpc_server.py"

def main():
    try:
        while True:
            p = subprocess.Popen([THREEDSMAX_PATH, RPC_SERVER_PATH])
            p.wait()
    except KeyboardInterrupt:
        killer = subprocess.Popen(['taskkill', '/F', '/T', '/PID',  str(p.pid)])
        killer.wait()

if __name__ == "__main__":
    main()