import numpy as np


N = 50
dw = 1
data = np.arange(50)
dd = 0.0
for i in range(N):
    dd += data[i]
    if i >= dw:
        avg = np.mean(data[i - dw:i + 1])
        print("Inx[{}] - Py Avg: {} ||| Avg = {}".format(i, avg, dd/(i + 1)))
        print(data[i - dw:i + 1])

# conv_avg = np.convolve(data, np.ones(dw)/dw, mode='valid')
# print(conv_avg)
# print(len(conv_avg))


print(np.insert(data, 0, 0))