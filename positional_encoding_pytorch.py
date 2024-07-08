import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class PositionalEncoding(nn.Module):
    """Positional encoding.
    This class is modified from https://github.com/JayParks/transformer/blob/master/transformer/modules.py.
    """

    def __init__(self, model_dim=512, max_seq_len=50):
       
        super(PositionalEncoding, self).__init__()

        # j//2 because we have sin and cos tow channels
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / model_dim) for j in range(model_dim)]
            for pos in range(max_seq_len)])
        # 0::2 表示从第0列开始，每隔两列取一个数，直到渠道最后为止
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # 位置编码positional_enc为0时，其代表padding的单词的位置编码，其中的维度索引全部为0
        pad_row = torch.zeros([1, model_dim])
        position_encoding = torch.FloatTensor(np.concatenate([pad_row, position_encoding]).astype(np.float32))

        # additional PAD position index
        self.position_encoding = nn.Embedding(max_seq_len + 1, model_dim)

        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):      
        # max(B,seq_len)，找出最长句子
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
    
        all_list = []
        for len in tqdm(input_len):
            a = np.asarray(list(range(1, len + 1)))  # 3,7
            # 这里从1开始的原因是0代表了padding的单词
            b = np.zeros((1, max_len - len), dtype=np.int8)[0]
            all_list.append(np.append(a, b))
        in_pos = tensor(all_list)
        return self.position_encoding(in_pos)

# print(PositionalEncoding()(torch.tensor([1,2])).shape)


test_sentences =  [[3,2,1,7,5,3,2,1,7,5,3,2,1,7,5],[4,2,6,3,2]]
test_sentences_len = torch.LongTensor([len(test_sentences[0]),len(test_sentences[1])])
print(test_sentences_len)
PE = PositionalEncoding(model_dim=64)
pe_enc_out = PE(test_sentences_len)
print(pe_enc_out.size())

plt.figure(figsize=(16, 5))
# [::-1]代表反向遍历
plt.plot(np.arange(15), pe_enc_out[0, :, :].data.numpy()[::-1])
plt.ylabel("value")
plt.xlabel("sequence_length")

plt.show()

print(pe_enc_out[1,0:5])

