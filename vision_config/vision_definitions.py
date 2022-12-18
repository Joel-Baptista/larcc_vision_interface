import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
USERNAME = os.getlogin()


def exports():       # Set CUDA and CUPTI paths
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH']= '/usr/local/cuda/bin:$PATH'
    os.environ['CPATH'] = '/usr/local/cuda/include:$CPATH'
    os.environ['LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LIBRARY_PATH'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
