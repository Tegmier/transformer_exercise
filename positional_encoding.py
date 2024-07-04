sentence = "I want to study attention mechanism"

import matplotlib.pyplot as plt
import math
import numpy as np
def positional_enc(pos, i):
    return math.sin(pos / (10000**(2*i/50)) if i%2==0 else math.cos(pos / (10000**(2*i/50))) )

def euclidean_distance(vector1, vector2):
    """
    计算两个向量之间的欧式距离。
    
    :param vector1: 第一个向量，列表或numpy数组
    :param vector2: 第二个向量，列表或numpy数组
    :return: 欧式距离
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    distance = np.linalg.norm(vector1 - vector2)
    return distance

def difference(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return vector1 - vector2

positional_ec = []
plt.figure(figsize=(10,5))
for pos_value in range(3):
    embedding_idxs = list(range(0,512))
    # print(embedding_idxs)
    sin_values = [positional_enc(pos_value, i) for i in embedding_idxs]
    positional_ec.append(sin_values)

print(euclidean_distance(positional_ec[0], positional_ec[1]))
ans = difference(positional_ec[0], positional_ec[1])
print(ans.shape)
    # plt.plot(embedding_idxs,sin_values,label=str(pos_value))
# plt.xlabel("embedding_idx")
# plt.ylabel("sin_value")
# plt.legend(loc = 'upper right')
# plt.show()
