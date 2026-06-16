#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

extern "C" __global__ AICORE void topk_gate_kernel_kernel(__gm__ uint8_t *scores_handle_raw, __gm__ uint8_t *topk_idx_handle_raw, int64_t num_tokens, uint64_t ffts_Addr) {
  __gm__ float *scores_handle = reinterpret_cast<__gm__ float *>(scores_handle_raw);
  __gm__ int *topk_idx_handle = reinterpret_cast<__gm__ int *>(topk_idx_handle_raw);
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 1, 32, 1, 32> scores_ub;
  TASSIGN(scores_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 2> topk_dst_ub;
  TASSIGN(topk_dst_ub, 128);
  tl::ascend_pto::TileUbDataND<float, 1, 192, 1, 192> tmp_ub_1;
  TASSIGN(tmp_ub_1, 160);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> topk_index_f32;
  TASSIGN(topk_index_f32, 928);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 32, 1, 1> tmp_ub;
  TASSIGN(tmp_ub, 960);
  tl::ascend_pto::TileUbDataND<int, 1, 8, 1, 1> topk_idx_out_ub;
  TASSIGN(topk_idx_out_ub, 992);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 1, 32, 1, 1, -1, 4, 1, 1, 32, pto::PadValue::Null>(scores_handle + (min(((cid * 2) + vid), (num_tokens - 1)) * 4), pto::Shape<1, 1, 1, 1, 32>(), pto::Stride<1, 1, -1, 4, 1>((num_tokens * 4)), 0, 0, 1, 4);

  for (int32_t i = 0; i < 28; ++i) {
      pipe_barrier(PIPE_ALL);
      scores_ub.SetValue((i + 4), -1.000000e+10f);
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    tl::ascend_pto::Sort<float, 32, 32, 1>(128 + ((0) * 4), 0 + ((0) * 4), 160 + ((0) * 4));
    pipe_barrier(PIPE_V);
    TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1>, tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 2>, MaskPattern::P1010>(topk_index_f32, topk_dst_ub);
    pipe_barrier(PIPE_V);
    tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> topk_index_f32_temp_0;
    TASSIGN(topk_index_f32_temp_0, 928 + 0 * 4);
    tl::ascend_pto::TileUbDataND<int, 1, 8, 1, 1> topk_idx_out_ub_temp_0;
    TASSIGN(topk_idx_out_ub_temp_0, 992 + 0 * 4);
    TCVT(topk_idx_out_ub_temp_0, topk_index_f32_temp_0, RoundMode::CAST_ROUND);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 1, 1, 1, 8>(topk_idx_handle + min(((cid * 2) + vid), (num_tokens - 1)), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 1, 1>(num_tokens), 992, 0, 1, 1);
#endif
}

extern "C" void call(uint8_t *scores_handle, uint8_t *topk_idx_handle, int64_t num_tokens, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    topk_gate_kernel_kernel<<<((num_tokens + 1) / 2), nullptr, stream>>>(scores_handle, topk_idx_handle, num_tokens, fftsAddr);
}
