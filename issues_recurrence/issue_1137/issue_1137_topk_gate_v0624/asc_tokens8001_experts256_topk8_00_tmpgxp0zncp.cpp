#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void topk_gate_kernel_kernel( GM_ADDR scores_handle,  GM_ADDR topk_idx_handle, int64_t num_tokens, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> scores;
  scores.SetGlobalBuffer((__gm__ float*)scores_handle);
  AscendC::GlobalTensor<int> topk_idx;
  topk_idx.SetGlobalBuffer((__gm__ int*)topk_idx_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto scores_ub_ping = ascend_ub.GetWithOffset<float>(256, 0);
  auto scores_ub_pong = ascend_ub.GetWithOffset<float>(256, 1024);
  auto topk_dst_ub_ping = ascend_ub.GetWithOffset<float>(512, 2048);
  auto tmp_ub_1 = ascend_ub.GetWithOffset<float>(2048, 4096);
  auto topk_index_f32_ping = ascend_ub.GetWithOffset<float>(256, 12288);
  auto topk_idx_out_ub_ping = ascend_ub.GetWithOffset<int>(8, 13312);
  auto topk_dst_ub_pong = ascend_ub.GetWithOffset<float>(512, 13344);
  auto topk_index_f32_pong = ascend_ub.GetWithOffset<float>(256, 15392);
  auto topk_idx_out_ub_pong = ascend_ub.GetWithOffset<int>(8, 16416);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    AscendC::PipeBarrier<PIPE_ALL>();
    if (0 < (((num_tokens + 1) / 2) - cid)) {
      tl::ascend::copy_gm_to_ub<float, 256>(scores_ub_ping[0], scores[(min(((cid * 2) + vid), (num_tokens - 1)) * 256)], (num_tokens * 256), 1, 256, -CUDART_INF_F);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::PipeBarrier<PIPE_ALL>();
      if (128 < (((num_tokens + 1) / 2) - cid)) {
        tl::ascend::copy_gm_to_ub<float, 256>(scores_ub_pong[0], scores[(min((((cid * 2) + vid) + 256), (num_tokens - 1)) * 256)], (num_tokens * 256), 1, 256, -CUDART_INF_F);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      tl::ascend::Sort<float>(topk_dst_ub_ping[0], scores_ub_ping[0], tmp_ub_1[0], 8, 256);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::GatherMask<float>(topk_index_f32_ping[0], topk_dst_ub_ping[0], 2);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(topk_idx_out_ub_ping[0], topk_index_f32_ping[0], AscendC::RoundMode::CAST_ROUND, 8);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
    if (256 < (((num_tokens + 1) / 2) - cid)) {
      for (int32_t i_step = 0; i_step < (((((num_tokens + 1) / 2) - cid) - 129) / 256); ++i_step) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
        tl::ascend::copy_gm_to_ub<float, 256>(scores_ub_ping[0], scores[(min(((((i_step * 512) + (cid * 2)) + vid) + 512), (num_tokens - 1)) * 256)], (num_tokens * 256), 1, 256, -CUDART_INF_F);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[(((i_step * 4096) + (cid * 16)) + (vid * 8))], topk_idx_out_ub_ping[0], (num_tokens * 8), ((1 <= (((num_tokens - vid) - (cid * 2)) - (i_step * 512))) ? 1 : 0), 8);
        tl::ascend::Sort<float>(topk_dst_ub_pong[0], scores_ub_pong[0], tmp_ub_1[0], 8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(topk_index_f32_pong[0], topk_dst_ub_pong[0], 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(topk_idx_out_ub_pong[0], topk_index_f32_pong[0], AscendC::RoundMode::CAST_ROUND, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        tl::ascend::copy_gm_to_ub<float, 256>(scores_ub_pong[0], scores[(min(((((i_step * 512) + (cid * 2)) + vid) + 768), (num_tokens - 1)) * 256)], (num_tokens * 256), 1, 256, -CUDART_INF_F);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
        AscendC::PipeBarrier<PIPE_MTE3>();
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[((((i_step * 4096) + (cid * 16)) + (vid * 8)) + 2048)], topk_idx_out_ub_pong[0], (num_tokens * 8), ((257 <= (((num_tokens - vid) - (cid * 2)) - (i_step * 512))) ? 1 : 0), 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
        tl::ascend::Sort<float>(topk_dst_ub_ping[0], scores_ub_ping[0], tmp_ub_1[0], 8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(topk_index_f32_ping[0], topk_dst_ub_ping[0], 2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(topk_idx_out_ub_ping[0], topk_index_f32_ping[0], AscendC::RoundMode::CAST_ROUND, 8);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((((((num_tokens + 1) / 2) + 127) - cid) % 256) / 128) == 1) {
        AscendC::PipeBarrier<PIPE_ALL>();
        tl::ascend::copy_gm_to_ub<float, 256>(scores_ub_ping[0], scores[(min((((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256) + (cid * 2)) + vid) - 256), (num_tokens - 1)) * 256)], (num_tokens * 256), 1, 256, -CUDART_INF_F);
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[(((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 2048) + (cid * 16)) + (vid * 8)) - 6144)], topk_idx_out_ub_ping[0], (num_tokens * 8), ((-767 <= (((num_tokens - vid) - (cid * 2)) - ((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256))) ? 1 : 0), 8);
        tl::ascend::Sort<float>(topk_dst_ub_pong[0], scores_ub_pong[0], tmp_ub_1[0], 8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(topk_index_f32_pong[0], topk_dst_ub_pong[0], 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(topk_idx_out_ub_pong[0], topk_index_f32_pong[0], AscendC::RoundMode::CAST_ROUND, 8);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
    if (128 < (((num_tokens + 1) / 2) - cid)) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if ((((((((num_tokens + 1) / 2) + 127) - cid) / 128) + 1) % 2) == 0) {
        tl::ascend::Sort<float>(topk_dst_ub_ping[0], scores_ub_ping[0], tmp_ub_1[0], 8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(topk_index_f32_ping[0], topk_dst_ub_ping[0], 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(topk_idx_out_ub_ping[0], topk_index_f32_ping[0], AscendC::RoundMode::CAST_ROUND, 8);
      } else {
        tl::ascend::Sort<float>(topk_dst_ub_pong[0], scores_ub_pong[0], tmp_ub_1[0], 8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(topk_index_f32_pong[0], topk_dst_ub_pong[0], 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(topk_idx_out_ub_pong[0], topk_index_f32_pong[0], AscendC::RoundMode::CAST_ROUND, 8);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
    if (129 <= (((num_tokens + 1) / 2) - cid)) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (((((((num_tokens + 1) / 2) + 127) - cid) % 256) / 128) == 0) {
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[(((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 2048) + (cid * 16)) + (vid * 8)) - 4096)], topk_idx_out_ub_ping[0], (num_tokens * 8), ((-511 <= (((num_tokens - vid) - (cid * 2)) - ((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256))) ? 1 : 0), 8);
      } else {
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[(((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 2048) + (cid * 16)) + (vid * 8)) - 4096)], topk_idx_out_ub_pong[0], (num_tokens * 8), ((-511 <= (((num_tokens - vid) - (cid * 2)) - ((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256))) ? 1 : 0), 8);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::PipeBarrier<PIPE_ALL>();
    if (1 <= (((num_tokens + 1) / 2) - cid)) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if ((((((((num_tokens + 1) / 2) + 127) - cid) / 128) + 1) % 2) == 0) {
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[(min((((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256) + (cid * 2)) + vid) - 256), (num_tokens - 1)) * 8)], topk_idx_out_ub_ping[0], (num_tokens * 8), 1, 8);
      } else {
        tl::ascend::copy_ub_to_gm<int, 8>(topk_idx[(min((((((((((num_tokens + 1) / 2) + 127) - cid) / 128) * 256) + (cid * 2)) + vid) - 256), (num_tokens - 1)) * 8)], topk_idx_out_ub_pong[0], (num_tokens * 8), 1, 8);
      }
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }
}

void topk_gate_kernel_kernel_tiling(int64_t num_tokens) {
}

extern "C" void call(uint8_t* scores_handle, uint8_t* topk_idx_handle, int64_t num_tokens, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  topk_gate_kernel_kernel_tiling(num_tokens);
  topk_gate_kernel_kernel<<<128, nullptr, stream>>>(scores_handle, topk_idx_handle, num_tokens, fftsAddr);
}
