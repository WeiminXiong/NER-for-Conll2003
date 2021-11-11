import torch
from torch.functional import Tensor


class Viterbi:
    def __init__(self, s_to_idx, v_to_idx, tran_martrix, emit_martrix):
        self.s_to_idx = s_to_idx
        self.v_to_idx = v_to_idx
        self.tran_martrix = torch.tensor(tran_martrix).transpose(1, 0)
        self.emit_martrix = torch.tensor(emit_martrix).transpose(1, 0)
        self.s_len = len(self.s_to_idx)

    def forward(self, ini_state, obs_sq):
        state_sq = [self.v_to_idx[i] for i in obs_sq]
        # 初始状态
        back_index = []
        ini_state = torch.tensor(ini_state)
        emit = self.emit_martrix[state_sq[0]]  # 初始状态的scores
        res = Tensor.log(ini_state) + Tensor.log(emit)

        for i in range(1, len(state_sq)):
            res_mid = []
            back = []
            for j in range(self.s_len):
                # 转移到j状态
                mid = res + Tensor.log(self.tran_martrix[j]) + Tensor.log(self.emit_martrix[state_sq[i], j])
                max_id = mid.argmax()
                res_mid.append(mid[max_id])
                back.append(max_id)
            res = torch.stack(res_mid)
            back_index.append(back)

        max_id = res.argmax()
        self.back_index = back_index
        return torch.exp(res[max_id]), max_id

    def backward(self, max_id):
        back_seq = [max_id]
        self.back_index.reverse()
        for item in self.back_index:
            max_id = item[max_id]
            back_seq.append(max_id)
        keys = list(self.s_to_idx.keys())
        back_seq.reverse()
        return [keys[i] for i in back_seq]


def toInd(lis):
    return {st: i for i, st in enumerate(lis)}


def __main__():
    state = ["健康", "发烧"]
    observation = ["正常", "冷", "头晕"]

    tran = [[0.7, 0.3], [0.4, 0.6]]
    emit = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]

    viterbi = Viterbi(toInd(state), toInd(observation), tran, emit)
    prob, max_id = viterbi.forward([0.6, 0.4], observation)
    print(viterbi.backward(max_id))
    print(prob)


__main__()
