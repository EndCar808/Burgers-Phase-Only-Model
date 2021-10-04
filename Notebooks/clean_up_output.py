import os
import sys
import shutil

# Dirs
data_dir = '/work/projects/TurbPhase/burgers_1d_code/Burgers_PO/Data'
output    = '/Output'

# Creat path
path = data_dir + output + '/Stats'

# Check if path is correct/exists
print(os.path.exists(path))

# Create list to hold undelted files
un_del_dirs = []
del_dirs    = []

# Loop through files in dir and try and delete
for f in os.listdir(path):
    try: 
        os.remove(path + '/' + f)
    except:
        try:
            os.rmdir(path + '/' + f)
        except:
            print("Unable to delete: {}".format(f))
            un_del_dirs.append(f)
        else:
            print("Deleted: {}".format(f))        
    else:
        print("Deleted: {}".format(f))


print()
print()
if not un_del_dirs:
    print("   Done!")
else:
    print("Undeleted files:")
    for i in un_del_dirs:
        print(i)