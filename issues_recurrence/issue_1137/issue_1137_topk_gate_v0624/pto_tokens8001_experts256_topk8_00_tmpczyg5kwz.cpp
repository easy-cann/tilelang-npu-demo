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

  tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> scores_ub_ping;
  TASSIGN(scores_ub_ping, 0);
  tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> scores_ub_pong;
  TASSIGN(scores_ub_pong, 1024);
  tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512> topk_dst_ub_ping;
  TASSIGN(topk_dst_ub_ping, 2048);
  tl::ascend_pto::TileUbDataND<float, 1, 1024, 1, 1024> tmp_ub_1;
  TASSIGN(tmp_ub_1, 4096);
  tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> topk_index_f32_ping;
  TASSIGN(topk_index_f32_ping, 8192);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 32, 1, 1> tmp_ub;
  TASSIGN(tmp_ub, 9216);
  tl::ascend_pto::TileUbDataND<int, 1, 8, 1, 8> topk_idx_out_ub_ping;
  TASSIGN(topk_idx_out_ub_ping, 9248);
  tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512> topk_dst_ub_pong;
  TASSIGN(topk_dst_ub_pong, 9280);
  tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> topk_index_f32_pong;
  TASSIGN(topk_index_f32_pong, 11328);
  tl::ascend_pto::TileUbDataND<int, 1, 8, 1, 8> topk_idx_out_ub_pong;
  TASSIGN(topk_idx_out_ub_pong, 12352);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    pipe_barrier(PIPE_ALL);
    if (0 < (((num_tokens + 1) / 2) - cid)) {
      tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 1, 256, 1, 1, -1, 256, 1, 1, 256, pto::PadValue::Min>(scores_handle + (min(((cid * 2) + vid), (num_tokens - 1)) * 256), pto::Shape<1, 1, 1, 1, 256>(), pto::Stride<1, 1, -1, 256, 1>((num_tokens * 256)), 0, 0, 1, 256);
      tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      pipe_barrier(PIPE_ALL);
      if (128 < (((num_tokens + 1) / 2) - cid)) {
        tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 1, 256, 1, 1, -1, 256, 1, 1, 256, pto::PadValue::Min>(scores_handle + (min((((cid * 2) + vid) + 256), (num_tokens - 1)) * 256), pto::Shape<1, 1, 1, 1, 256>(), pto::Stride<1, 1, -1, 256, 1>((num_tokens * 256)), 1024, 0, 1, 256);
      }
      pipe_barrier(PIPE_ALL);
      tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> (0);
      tl::ascend_pto::Sort<float, 256, 256, -1>(2048 + ((0) * 4), 0 + ((0) * 4), 4096 + ((0) * 4));
      pipe_barrier(PIPE_V);
      TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256>, tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512>, MaskPattern::P1010>(topk_index_f32_ping, topk_dst_ub_ping);
      pipe_barrier(PIPE_V);
      TCVT(topk_idx_out_ub_ping, topk_index_f32_ping, RoundMode::CAST_ROUND);
    }
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
    if (256 < (((num_tokens + 1) / 2) - cid)) {

  for (int32_t i_step = 0; i_step < (((((num_tokens + 1) / 2) - cid) - 129) / 256); ++i_step) {
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 1, 256, 1, 1, -1, 256, 1, 1, 256, pto::PadValue::Min>(scores_handle + (min(((((i_step * 512) + (cid * 2)) + vid) + 512), (num_tokens - 1)) * 256), pto::Shape<1, 1, 1, 1, 256>(), pto::Stride<1, 1, -1, 256, 1>((num_tokens * 256)), 0, 0, 1, 256);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + (((i_step * 4096) + (cid * 16)) + (vid * 8)), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 9248, 0, ((1 <= (((num_tokens - vid) - (cid * 2)) - (i_step * 512))) ? 1 : 0), 8);
        tl::ascend_pto::Sort<float, 256, 256, -1>(9280 + ((0) * 4), 1024 + ((0) * 4), 4096 + ((0) * 4));
        pipe_barrier(PIPE_V);
        TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256>, tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512>, MaskPattern::P1010>(topk_index_f32_pong, topk_dst_ub_pong);
        pipe_barrier(PIPE_V);
        TCVT(topk_idx_out_ub_pong, topk_index_f32_pong, RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 1, 256, 1, 1, -1, 256, 1, 1, 256, pto::PadValue::Min>(scores_handle + (min(((((i_step * 512) + (cid * 2)) + vid) + 768), (num_tokens - 1)) * 256), pto::Shape<1, 1, 1, 1, 256>(), pto::Stride<1, 1, -1, 256, 1>((num_tokens * 256)), 1024, 0, 1, 256);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        pipe_barrier(PIPE_MTE3);
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + ((((i_step * 4096) + (cid * 16)) + (vid * 8)) + 2048), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 12352, 0, ((257 <= (((num_tokens - vid) - (cid * 2)) - (i_step * 512))) ? 1 : 0), 8);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        tl::ascend_pto::Sort<float, 256, 256, -1>(2048 + ((0) * 4), 0 + ((0) * 4), 4096 + ((0) * 4));
        pipe_barrier(PIPE_V);
        TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256>, tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512>, MaskPattern::P1010>(topk_index_f32_ping, topk_dst_ub_ping);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        pipe_barrier(PIPE_V);
        TCVT(topk_idx_out_ub_ping, topk_index_f32_ping, RoundMode::CAST_ROUND);
      }
      pipe_barrier(PIPE_ALL);
      if (((((((num_tokens + 1) / 2) + 127) - cid) % 256) / 128) == 1) {
        pipe_barrier(PIPE_ALL);
        tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 1, 256, 1, 1, -1, 256, 1, 1, 256, pto::PadValue::Min>(scores_handle + (min((((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256) + (cid * 2)) + vid) - 256), (num_tokens - 1)) * 256), pto::Shape<1, 1, 1, 1, 256>(), pto::Stride<1, 1, -1, 256, 1>((num_tokens * 256)), 0, 0, 1, 256);
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + (((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 2048) + (cid * 16)) + (vid * 8)) - 6144), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 9248, 0, ((-767 <= (((num_tokens - vid) - (cid * 2)) - ((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256))) ? 1 : 0), 8);
        tl::ascend_pto::Sort<float, 256, 256, -1>(9280 + ((0) * 4), 1024 + ((0) * 4), 4096 + ((0) * 4));
        pipe_barrier(PIPE_V);
        TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256>, tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512>, MaskPattern::P1010>(topk_index_f32_pong, topk_dst_ub_pong);
        pipe_barrier(PIPE_V);
        TCVT(topk_idx_out_ub_pong, topk_index_f32_pong, RoundMode::CAST_ROUND);
      }
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
    if (128 < (((num_tokens + 1) / 2) - cid)) {
      pipe_barrier(PIPE_ALL);
      if ((((((((num_tokens + 1) / 2) + 127) - cid) / 128) + 1) % 2) == 0) {
        tl::ascend_pto::Sort<float, 256, 256, -1>(2048 + ((0) * 4), 0 + ((0) * 4), 4096 + ((0) * 4));
        pipe_barrier(PIPE_V);
        TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256>, tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512>, MaskPattern::P1010>(topk_index_f32_ping, topk_dst_ub_ping);
        pipe_barrier(PIPE_V);
        TCVT(topk_idx_out_ub_ping, topk_index_f32_ping, RoundMode::CAST_ROUND);
      } else {
        tl::ascend_pto::Sort<float, 256, 256, -1>(9280 + ((0) * 4), 1024 + ((0) * 4), 4096 + ((0) * 4));
        pipe_barrier(PIPE_V);
        TGATHER<tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256>, tl::ascend_pto::TileUbDataND<float, 1, 512, 1, 512>, MaskPattern::P1010>(topk_index_f32_pong, topk_dst_ub_pong);
        pipe_barrier(PIPE_V);
        TCVT(topk_idx_out_ub_pong, topk_index_f32_pong, RoundMode::CAST_ROUND);
      }
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
    if (129 <= (((num_tokens + 1) / 2) - cid)) {
      pipe_barrier(PIPE_ALL);
      if (((((((num_tokens + 1) / 2) + 127) - cid) % 256) / 128) == 0) {
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + (((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 2048) + (cid * 16)) + (vid * 8)) - 4096), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 9248, 0, ((-511 <= (((num_tokens - vid) - (cid * 2)) - ((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256))) ? 1 : 0), 8);
      } else {
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + (((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 2048) + (cid * 16)) + (vid * 8)) - 4096), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 12352, 0, ((-511 <= (((num_tokens - vid) - (cid * 2)) - ((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256))) ? 1 : 0), 8);
      }
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
    if (1 <= (((num_tokens + 1) / 2) - cid)) {
      pipe_barrier(PIPE_ALL);
      if ((((((((num_tokens + 1) / 2) + 127) - cid) / 128) + 1) % 2) == 0) {
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + (min((((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256) + (cid * 2)) + vid) - 256), (num_tokens - 1)) * 8), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 9248, 0, 1, 8);
      } else {
        tl::ascend_pto::copy_ub_to_gm_dynamic<int, int, 1, 1, 1, 1, 8, 1, 1, -1, 8, 1, 1, 8>(topk_idx_handle + (min((((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256) + (cid * 2)) + vid) - 256), (num_tokens - 1)) * 8), pto::Shape<1, 1, 1, 1, 8>(), pto::Stride<1, 1, -1, 8, 1>((num_tokens * 8)), 12352, 0, 1, 8);
      }
      pipe_barrier(PIPE_ALL);
    }
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" void call(uint8_t *scores_handle, uint8_t *topk_idx_handle, int64_t num_tokens, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    topk_gate_kernel_kernel<<<128, nullptr, stream>>>(scores_handle, topk_idx_handle, num_tokens, fftsAddr);
}
