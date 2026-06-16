import tilelang
import tilelang.language as T
import torch
import numpy as np
import pytest
import os

tilelang.cache.clear_cache()
tilelang.disable_cache()

# ===================== 补齐缺失的 3 个工具函数 =====================
def generate_test_params(is_benchmark: bool = False):
    """生成测试用例参数组合：num_tokens, num_experts, num_topk"""
    test_cases = []
    # 常规小用例、边界用例、对齐/非对齐专家数、向量分组边界
    if is_benchmark:
        # 压测大尺寸
        cases = [
            {"num_tokens": 1024, "num_experts": 8, "num_topk": 2},
            # {"num_tokens": 4096, "num_experts": 16, "num_topk": 4},
            # {"num_tokens": 8192, "num_experts": 32, "num_topk": 8},
            # {"num_tokens": 12288, "num_experts": 64, "num_topk": 8},
        ]
    else:
        # 单元测试小用例 + 边界用例
        cases = [
            {"num_tokens": 1, "num_experts": 4, "num_topk": 1},
            # {"num_tokens": 5, "num_experts": 8, "num_topk": 2},
            # {"num_tokens": 32, "num_experts": 10, "num_topk": 3},
            # {"num_tokens": 63, "num_experts": 32, "num_topk": 4},
            # {"num_tokens": 128, "num_experts": 12, "num_topk": 6},
        ]
    for case in cases:
        test_cases.append(case)
    return test_cases


def make_param_id(params: dict) -> str:
    """为 pytest 参数生成可读性 ID，用于测试报告展示"""
    nt = params["num_tokens"]
    ne = params["num_experts"]
    nk = params["num_topk"]
    return f"tokens={nt}_experts={ne}_topk={nk}"


def generate_test_data(params: dict) -> torch.Tensor:
    """根据参数生成测试 scores 张量，默认放 NPU/CPU，float32、连续内存"""
    num_tokens = params["num_tokens"]
    num_experts = params["num_experts"]
    # 固定随机种子，保证结果可复现
    torch.manual_seed(42)
    # 生成 [-10, 10] 随机分数，保证 topk 结果稳定可比对
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32) * 20 - 10
    # 适配昇腾 NPU / CPU，自动匹配设备
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        dev = "npu"
    else:
        dev = "cpu"
    scores = scores.to(dev).contiguous()
    return scores

# ===================================================================
pass_configs = {
    tilelang.PassConfigKey.TIR_MERGE_STATIC_SMEM: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


# @tilelang.jit(pass_configs=pass_configs, target="pto")
@tilelang.jit(pass_configs=pass_configs)
def get_topk_gate_kernel(num_experts: int, num_topk: int):
    num_tokens = T.symbolic('num_tokens')
    num_threads = 32
    num_aligned_experts = (num_experts + num_threads - 1) // num_threads * num_threads

    VEC_NUM = 2
    num_batches = (num_tokens + VEC_NUM - 1) // VEC_NUM
    num_blocks = 128

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), "float32"],
        topk_idx: T.Tensor[(num_tokens, num_topk), "int32"],
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            scores_ub_ping = T.alloc_ub((num_aligned_experts,), "float32")
            scores_ub_pong = T.alloc_ub((num_aligned_experts,), "float32")
            topk_idx_out_ub_ping = T.alloc_ub((num_topk,), "int32")
            topk_idx_out_ub_pong = T.alloc_ub((num_topk,), "int32")
            topk_dst_ub = T.alloc_ub((2 * num_topk,), "float32")
            topk_index_f32 = T.alloc_ub((num_topk,), "float32")

            num_total_iters = (num_batches + num_blocks - 1) // num_blocks
            for i in T.serial(num_total_iters):
                batch_idx = cid + i * num_blocks
                if batch_idx < num_batches:
                    token_idx = T.min(batch_idx * VEC_NUM + vid, num_tokens - 1)

                    # 默认赋值为 ping，保证解析器识别变量定义
                    cur_scores = scores_ub_ping
                    cur_out    = topk_idx_out_ub_ping
                    nxt_scores = scores_ub_pong
                    nxt_out    = topk_idx_out_ub_pong
                    # 根据奇偶性覆盖为 pong 组合
                    if i % 2 == 1:
                        cur_scores = scores_ub_pong
                        cur_out    = topk_idx_out_ub_pong
                        nxt_scores = scores_ub_ping
                        nxt_out    = topk_idx_out_ub_ping

                    T.copy(scores[token_idx, 0:num_experts], cur_scores[0:num_experts])
                    for j in range(num_aligned_experts - num_experts):
                        cur_scores[num_experts + j] = -1e10

                    T.tile.topk(topk_dst_ub, cur_scores, num_topk, num_aligned_experts)
                    T.tile.gather_mask(topk_index_f32, topk_dst_ub, "P1010")
                    T.tile.cast(cur_out, topk_index_f32, "CAST_ROUND", num_topk)

                    T.copy(cur_out[0:num_topk], topk_idx[token_idx, 0:num_topk])

                    next_batch_idx = batch_idx + num_blocks
                    if next_batch_idx < num_batches:
                        next_token_idx = T.min(next_batch_idx * VEC_NUM + vid, num_tokens - 1)
                        T.copy(scores[next_token_idx, 0:num_experts], nxt_scores[0:num_experts])

    return topk_gate_kernel


def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts
    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int32, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    kernel = get_topk_gate_kernel(num_experts, num_topk)

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    kernel(scores, topk_idx)
    return topk_idx


def stable_topk(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    _, sorted_indices = torch.sort(scores, dim=1, descending=True, stable=True)
    return sorted_indices[:, :num_topk].to(torch.int32).contiguous()


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_topk_gate(params):
    scores = generate_test_data(params)
    num_topk = params['num_topk']

    topk_idx_ref = stable_topk(scores, num_topk)
    topk_idx = topk_gate(scores, num_topk)
    # 设备对齐后再比对
    np.testing.assert_equal(topk_idx.cpu().numpy(), topk_idx_ref.cpu().numpy())
    print(f"Test passed for params: {make_param_id(params)}")


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])