import os
import pytest
import numpy as np
import torch
import tilelang
import tilelang.language as T

pass_configs = {
  tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
  tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True
}

@tilelang.jit(pass_configs=pass_configs, target="pto")
# @tilelang.jit(pass_configs=pass_configs)
def get_topk_gate_kernel(num_experts: int, num_topk: int):
    num_tokens = T.symbolic('num_tokens')
    num_threads = 32
    num_aligned_experts = (num_experts + num_threads - 1) // num_threads * num_threads
    
    VEC_NUM = 2
    num_batches = (num_tokens + VEC_NUM - 1) // VEC_NUM
    num_blocks = 128
    
    aligned_num_topk_f32 = (num_topk + 7) // 8 * 8
    aligned_num_topk_i64 = (num_topk + 3) // 4 * 4

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), "float32"],
        topk_idx: T.Tensor[(num_tokens, num_topk), "int32"],
    ):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            scores_ub_ping = T.alloc_ub((num_aligned_experts,), "float32")
            scores_ub_pong = T.alloc_ub((num_aligned_experts,), "float32")

            topk_idx_out_ub_ping = T.alloc_ub((aligned_num_topk_i64,), "int32")
            topk_idx_out_ub_pong = T.alloc_ub((aligned_num_topk_i64,), "int32")

            topk_dst_ub_ping = T.alloc_ub((2 * num_aligned_experts,), "float32")
            topk_dst_ub_pong = T.alloc_ub((2 * num_aligned_experts,), "float32")
            
            topk_index_f32_ping = T.alloc_ub((num_aligned_experts,), "float32")
            topk_index_f32_pong = T.alloc_ub((num_aligned_experts,), "float32")

            my_iters = (num_batches -cid + num_blocks - 1) // num_blocks
            
            # === Prologue ===
            # 优化：合并条件判断嵌套
            if my_iters > 0:
                batch_idx_0 = cid
                token_idx_0 = T.min(batch_idx_0 * VEC_NUM + vid, num_tokens - 1)

                T.copy(scores[token_idx_0, 0:num_experts], scores_ub_ping, pad_value=-T.infinity("float32"))
                T.set_flag("mte2", "v", 0)
              
            if my_iters > 1:
                batch_idx_1 = cid + num_blocks
                token_idx_1 = T.min(batch_idx_1 * VEC_NUM + vid, num_tokens - 1)

                T.copy(scores[token_idx_1, 0:num_experts], scores_ub_pong, pad_value=-T.infinity("float32"))
                T.wait_flag("mte2", "v", 0)
              
            T.tile.sort(topk_dst_ub_ping, scores_ub_ping, num_experts)
            # T.pipe_barrier("V")

            T.tile.gather_mask(topk_index_f32_ping, topk_dst_ub_ping, "P1010")

            T.tile.cast(topk_idx_out_ub_ping, topk_index_f32_ping, "CAST_ROUND", num_topk)
            
            # === Main Loop ===
            # 优化：步长为2循环展开，cedilla消除流水线中的if i % 2 == 1 分支
            loop_iters = my_iters - 2
            if loop_iters > 0:
                for i_step in T.serial(loop_iters // 2):
                    # --- Ping Phase ---
                    i_ping = i_step * 2 + 1
                    
                    batch_idx_prev_ping = cid + (i_ping - 1) * num_blocks
                    batch_idx_next_ping = cid + (i_ping + 1) * num_blocks
                    
                    token_idx_prev_ping = batch_idx_prev_ping * VEC_NUM + vid
                    token_idx_next_ping = T.min(batch_idx_next_ping * VEC_NUM + vid, num_tokens - 1)
                    
                    # T.pipe_barrier("V")
                    T.barrier_all()
                
                    T.copy(scores[token_idx_next_ping, 0:num_experts], scores_ub_ping, pad_value=-T.infinity("float32"))
                    T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_prev_ping, 0:num_topk])
                    
                    T.tile.sort(topk_dst_ub_pong, scores_ub_pong, num_experts)
                    # T.pipe_barrier("V")
                    T.tile.gather_mask(topk_index_f32_pong, topk_dst_ub_pong, "P1010")
                    T.tile.cast(topk_idx_out_ub_pong, topk_index_f32_pong, "CAST_ROUND", num_topk)
                    
                    # --- Pong Phase ---
                    i_pong = i_step * 2 + 2
                    
                    batch_idx_prev_pong = cid + (i_pong - 1) * num_blocks
                    batch_idx_next_pong = cid + (i_pong + 1) * num_blocks
                    
                    token_idx_prev_pong = batch_idx_prev_pong * VEC_NUM + vid
                    token_idx_next_pong = T.min(batch_idx_next_pong * VEC_NUM + vid, num_tokens - 1)
                    
                    # T.pipe_barrier("V")
                    T.barrier_all()
                    
                    T.copy(scores[token_idx_next_pong, 0:num_experts], scores_ub_pong, pad_value=-T.infinity("float32"))
                    T.copy(topk_idx_out_ub_pong[0:num_topk], topk_idx[token_idx_prev_pong, 0:num_topk])
                    
                    T.tile.sort(topk_dst_ub_ping, scores_ub_ping, num_experts)
                    # T.pipe_barrier("V")                
                    T.tile.gather_mask(topk_index_f32_ping, topk_dst_ub_ping, "P1010")
                    T.tile.cast(topk_idx_out_ub_ping, topk_index_f32_ping, "CAST_ROUND", num_topk)

            if loop_iters % 2 == 1:
                i_last = loop_iters
                batch_idx_prev_last = cid + (i_last - 1) * num_blocks
                batch_idx_next_last = cid + (i_last + 1) * num_blocks
                
                token_idx_prev_last = batch_idx_prev_last * VEC_NUM + vid
                token_idx_next_last = T.min(batch_idx_next_last * VEC_NUM + vid, num_tokens - 1)

                # T.pipe_barrier("V")
                T.barrier_all()
                
                T.copy(scores[token_idx_next_last, 0:num_experts], scores_ub_ping, pad_value=-T.infinity("float32"))
                T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_prev_last, 0:num_topk])
                
                T.tile.sort(topk_dst_ub_pong, scores_ub_pong, num_experts)
                # T.pipe_barrier("V")
                T.tile.gather_mask(topk_index_f32_pong, topk_dst_ub_pong, "P1010")
                T.tile.cast(topk_idx_out_ub_pong, topk_index_f32_pong, "CAST_ROUND", num_topk)
              
            # === Epilogue ===
            T.barrier_all()
            i_epi2 = my_iters - 1
            if i_epi2 > 0:
                if i_epi2 % 2 == 0:
                    T.tile.sort(topk_dst_ub_ping, scores_ub_ping, num_experts)
                    # T.pipe_barrier("V")
                    T.tile.gather_mask(topk_index_f32_ping, topk_dst_ub_ping, "P1010")
                    T.tile.cast(topk_idx_out_ub_ping, topk_index_f32_ping, "CAST_ROUND", num_topk)
                else:
                    T.tile.sort(topk_dst_ub_pong, scores_ub_pong, num_experts)
                    # T.pipe_barrier("V")
                    T.tile.gather_mask(topk_index_f32_pong, topk_dst_ub_pong, "P1010")
                    T.tile.cast(topk_idx_out_ub_pong, topk_index_f32_pong, "CAST_ROUND", num_topk)
                
            i_epi1 = my_iters - 2
            if i_epi1 >= 0:
                batch_idx_epi1 = cid + i_epi1 * num_blocks
                token_idx_epi1 = batch_idx_epi1 * VEC_NUM + vid
                if i_epi1 % 2 == 0:
                    T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_epi1, 0:num_topk])
                else:
                    T.copy(topk_idx_out_ub_pong[0:num_topk], topk_idx[token_idx_epi1, 0:num_topk])
            
            if i_epi2 >= 0:
                batch_idx_epi2 = cid + i_epi2 * num_blocks
                token_idx_epi2 = T.min(batch_idx_epi2 * VEC_NUM + vid, num_tokens - 1)
                # T.pipe_barrier("V")
                if i_epi2 % 2 == 0:
                    T.copy(topk_idx_out_ub_ping[0:num_topk], topk_idx[token_idx_epi2, 0:num_topk])
                else:
                    T.copy(topk_idx_out_ub_pong[0:num_topk], topk_idx[token_idx_epi2, 0:num_topk])
    
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


def generate_test_params():
    return [
        {"num_tokens": 1, "num_experts": 4, "num_topk": 1},
        # {"num_tokens": 5, "num_experts": 8, "num_topk": 2},
        # {"num_tokens": 32, "num_experts": 10, "num_topk": 3},
        # {"num_tokens": 63, "num_experts": 32, "num_topk": 4},
        # {"num_tokens": 128, "num_experts": 12, "num_topk": 6},
    ]


def make_param_id(params: dict) -> str:
    nt = params["num_tokens"]
    ne = params["num_experts"]
    nk = params["num_topk"]
    return f"tokens={nt}_experts={ne}_topk={nk}"


@pytest.mark.parametrize('params', generate_test_params(), ids=make_param_id)
def test_topk_gate(params):
    num_tokens = params['num_tokens']
    num_experts = params['num_experts']
    num_topk = params['num_topk']

    torch.manual_seed(42)
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32) * 20 - 10
    if hasattr(torch, "npu") and torch.npu.is_available():
        scores = scores.to("npu").contiguous()
    elif torch.cuda.is_available():
        scores = scores.to("cuda").contiguous()

    topk_idx_ref = stable_topk(scores, num_topk)
    topk_idx = topk_gate(scores, num_topk)

    np.testing.assert_equal(topk_idx.cpu().numpy(), topk_idx_ref.cpu().numpy())
    print(f"Test passed for params: {make_param_id(params)}")


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])