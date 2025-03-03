import megfile as mf
import platform
if platform.system() != "Windows":
    import fcntl
stat_info = mf.smart_listdir(r'D:\dataset\NTIRE_2025')
print(stat_info)

