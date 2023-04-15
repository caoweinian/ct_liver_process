import os
import platform

LiTSBase = os.path.join('dataset', 'medical_liver', 'LiTS')


def _get_prefix() -> str:
    sys_name = platform.uname()[0]
    if sys_name == 'Darwin':
        return os.path.join('/Volumes', 'LaCie_exFAT')
    elif sys_name == 'Windows':
        return 'E:\\'
    else:
        return os.path.join('/home', 'lab510', 'cwn')


LocalDirectory = os.path.join(_get_prefix(), LiTSBase)
TrainingSetLen = 131
TestingSetLen = 70
