# from random import sample
# import shutil
# import os
# for dir in os.listdir('Animals_with_Attributes2/JPEGImages/'):
#   path = os.path.join('Animals_with_Attributes2/JPEGImages/',dir)
#   temp = os.listdir(path)
#   lst = sample(temp, int(0.1 * len(temp)))
#   for f in temp:
#     if f in lst:
#       shutil.copy(os.path.join(path,f), 'test_dir/')
#     else:
#       shutil.copy(os.path.join(path,f), 'train_dir/')

# see dataset_sun.ipynb for SUN dataset


from random import sample
import shutil
import os
for dir in os.listdir('CUB/Data/'):
  path = os.path.join('CUB/Data/', dir)
  temp = os.listdir(path)
  lst = sample(temp, int(0.1 * len(temp)))
  for f in temp:
    if f in lst:
      shutil.copy(os.path.join(path,f), 'CUB/test_dir/' + dir + '@' + f)
    else:
      shutil.copy(os.path.join(path,f), 'CUB/train_dir/' + dir + '@' + f)