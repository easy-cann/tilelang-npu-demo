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
  auto nxt_scores = ascend_ub.GetWithOffset<float>(32, 0);
  auto topk_dst_ub = ascend_ub.GetWithOffset<float>(2, 128);
  auto tmp_ub_1 = ascend_ub.GetWithOffset<float>(128, 160);
  auto topk_index_f32 = ascend_ub.GetWithOffset<float>(1, 672);
  auto nxt_out = ascend_ub.GetWithOffset<int>(1, 704);
  auto cur_scores = ascend_ub.GetWithOffset<float>(32, 736);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    for (int32_t i = 0; i < ((num_tokens + 255) / 256); ++i) {
      if (((i * 128) + cid) < ((num_tokens + 1) / 2)) {
        tl::ascend::copy_gm_to_ub<float, 4>(nxt_scores[0], scores[(min((((i * 256) + (cid * 2)) + vid), (num_tokens - 1)) * 4)], (num_tokens * 4), 1, 4, 0.000000e+00f);
        for (int32_t j = 0; j < 28; ++j) {
          nxt_scores.SetValue((j + 4), -1.000000e+10f);
        }
        tl::ascend::TopK<float>(topk_dst_ub[0], nxt_scores[0], tmp_ub_1[0], 1, 1, 32);
        tl::ascend::GatherMask<float>(topk_index_f32[0], topk_dst_ub[0], 2);
        AscendC::Cast(nxt_out[0], topk_index_f32[0], AscendC::RoundMode::CAST_ROUND, 1);
        tl::ascend::copy_ub_to_gm<int, 1>(topk_idx[min((((i * 256) + (cid * 2)) + vid), (num_tokens - 1))], nxt_out[0], num_tokens, 1, 1);
        if ((((i * 128) + cid) + 128) < ((num_tokens + 1) / 2)) {
          tl::ascend::copy_gm_to_ub<float, 4>(cur_scores[0], scores[(min(((((i * 256) + (cid * 2)) + vid) + 256), (num_tokens - 1)) * 4)], (num_tokens * 4), 1, 4, 0.000000e+00f);
        }
      }
    }
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
