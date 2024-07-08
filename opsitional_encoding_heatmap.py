import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import math

def positional_enc(pos, i):
    return math.sin(pos / (10000**(2*i/50)) if i%2==0 else math.cos(pos / (10000**(2*i/50))) )

#定义热图的横纵坐标
pos_idxs = list(range(0,16,1)) # 纵坐标
embedding_idxs = list(range(50))# 横坐标
sin_values = [] # (30*50)
for pos_idx in range(len(pos_idxs)):
    temp = []
    for emb_idx in range(len(embedding_idxs)):
        value = positional_enc(pos_idx,emb_idx)
        temp.append(value)
    sin_values.append(temp)

print(len(sin_values))

#作图阶段
fig = plt.figure(figsize=(10,5))

#定义画布为1*1个划分，并在第1个位置上进行作图
ax = fig.add_subplot(111)

# #定义横纵坐标的刻度
# ax.set_xticks(range(len(embedding_idxs)))
# ax.set_xticklabels(embedding_idxs)
# ax.set_yticks(range(len(pos_idxs)))
# ax.set_yticklabels(pos_idxs)
# #作图并选择热图的颜色填充风格，这里选择hot
# im = ax.imshow(sin_values, cmap=plt.cm.hot_r)

plt.xticks(range(len(embedding_idxs)), embedding_idxs)
plt.yticks(range(len(pos_idxs)), pos_idxs)

# 作图并选择热图的颜色填充风格，这里选择hot
plt.imshow(sin_values, cmap=plt.cm.hot_r)


#增加右侧的颜色刻度条
# plt.colorbar(im)
plt.xlabel("embedding_idx")
plt.ylabel("pos_idx")
plt.show()
