// SPDX-License-Identifier: AGPL-3.0-only
// Standalone GB10 byte-parity gate for the exact DeepSeek W2A8 H128 emitters.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef W2A8_EMIT_PROBE_BUILD_ID
#error "W2A8_EMIT_PROBE_BUILD_ID must be supplied by the receipt-producing build"
#endif

#include "exl3_w2a8_h128_emit.cu"
#include "../common/per_token_group_quant_fp8.cu"

namespace probe {
constexpr unsigned kPreWidth = 4096, kPostWidth = 2048, kExperts = 3;
constexpr unsigned kMaxRows = 129, kTokens = 137;
constexpr std::size_t kGuardBytes = 256;
constexpr unsigned char kCanary = 0xcd;

[[noreturn]] void fail(const char* check_name) {
    std::fprintf(stderr, "FAIL check=%s\n", check_name);
    std::exit(1);
}
[[noreturn]] void cuda_fail(const char* op, cudaError_t error) {
    std::fprintf(stderr, "FAIL cuda=%s error=%s\n", op, cudaGetErrorString(error));
    std::exit(2);
}
void check(cudaError_t error, const char* op) {
    if (error != cudaSuccess) cuda_fail(op, error);
}

struct Buffer {
    unsigned char* base = nullptr;
    unsigned char* data = nullptr;
    std::size_t bytes = 0;
    explicit Buffer(std::size_t size, unsigned char fill_value = 0) : bytes(size) {
        check(cudaMalloc(&base, bytes + 2 * kGuardBytes), "cudaMalloc");
        data = base + kGuardBytes;
        check(cudaMemset(base, kCanary, bytes + 2 * kGuardBytes), "guard memset");
        check(cudaMemset(data, fill_value, bytes), "data memset");
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (base != nullptr) cudaFree(base); }
    template <typename T> void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) fail("upload-size");
        check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice), "upload");
    }
    void fill(unsigned char value) { check(cudaMemset(data, value, bytes), "fill"); }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> result(bytes);
        check(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost), "download");
        return result;
    }
    bool guards_clean() const {
        std::vector<unsigned char> guard(2 * kGuardBytes);
        check(cudaMemcpy(guard.data(), base, kGuardBytes, cudaMemcpyDeviceToHost), "prefix");
        check(cudaMemcpy(guard.data() + kGuardBytes, data + bytes, kGuardBytes,
                         cudaMemcpyDeviceToHost), "suffix");
        for (unsigned char value : guard) if (value != kCanary) return false;
        return true;
    }
};

std::uint16_t to_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}
std::vector<std::uint16_t> make_values(unsigned rows, unsigned width, unsigned salt) {
    static constexpr std::uint16_t edge[] = {
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x4120, 0xc120, 0x41a0, 0xc1a0,
        0x3eab, 0xbeab, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1,
    };
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * width);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned column = 0; column < width; ++column) {
            const std::size_t at = static_cast<std::size_t>(row) * width + column;
            if (column < 128) result[at] = (column & 1u) ? 0x8000 : 0x0000;
            else if (column < 128 + sizeof(edge) / sizeof(edge[0]))
                result[at] = edge[column - 128];
            else {
                const int raw = static_cast<int>((at * 37 + salt * 101) % 4093) - 2046;
                result[at] = to_bf16(static_cast<float>(raw) / 97.0f);
            }
        }
    }
    return result;
}
std::vector<std::uint16_t> make_signs(unsigned width, unsigned salt) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kExperts) * width);
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = ((i * 17 + (i >> 3) + ((salt >> (i % 5)) & 1u)) & 1u)
            ? 0xbc00u : 0x3c00u;
    return result;
}
std::vector<std::uint64_t> make_table(const Buffer& signs, unsigned width) {
    std::vector<std::uint64_t> result(kExperts);
    for (unsigned expert = 0; expert < kExperts; ++expert)
        result[expert] = reinterpret_cast<std::uintptr_t>(signs.data) +
            static_cast<std::uint64_t>(expert) * width * sizeof(std::uint16_t);
    return result;
}
std::uint64_t hash_bytes(std::uint64_t hash, const void* pointer, std::size_t bytes) {
    const auto* data = static_cast<const unsigned char*>(pointer);
    for (std::size_t i = 0; i < bytes; ++i) { hash ^= data[i]; hash *= 0x100000001b3ull; }
    return hash;
}
template <typename T> std::uint64_t hash_vector(std::uint64_t hash,
                                                const std::vector<T>& values) {
    return hash_bytes(hash, values.data(), values.size() * sizeof(T));
}

// BEGIN emitter routing contract
constexpr unsigned row_cases[] = {1, 31, 32, 33, 63, 64, 65, 129};
std::vector<std::int32_t> sorted_token_ids() {
    std::vector<std::int32_t> nonidentity(kMaxRows);
    for (unsigned row = 0; row < kMaxRows; ++row)
        nonidentity[row] = static_cast<std::int32_t>(row % 11 == 0 ? 3 : (row * 17) % kTokens);
    return nonidentity;  // Deliberate duplicate token ownership.
}
std::vector<std::int32_t> sorted_expert_ids() {
    std::vector<std::int32_t> duplicate(kMaxRows);
    for (unsigned row = 0; row < kMaxRows; ++row) duplicate[row] = (row * 5 + 1) % kExperts;
    return duplicate;
}
constexpr const int* identity_token_ids = nullptr;
// Device pointer tables: gate_suh_tab, up_suh_tab, gate_svh_tab, up_svh_tab,
// and down_suh_tab are independently signed and selected by sorted_expert_ids.
// END emitter routing contract

// BEGIN emitter probe kernel contract
void pre_reference(const __nv_bfloat16* input, const int* tokens, const int* experts,
                   const unsigned long long* gate_tab, const unsigned long long* up_tab,
                   __nv_bfloat16* gate, __nv_bfloat16* up, __nv_bfloat16* fixed_gate,
                   __nv_bfloat16* fixed_up, unsigned rows) {
    exl3_h128_pre_rows<<<dim3(rows,4,1),dim3(256,1,1)>>>(input,tokens,experts,gate_tab,gate,kPreWidth);
    exl3_h128_pre_rows<<<dim3(rows,4,1),dim3(256,1,1)>>>(input,tokens,experts,up_tab,up,kPreWidth);
    exl3_h128_pre_dual_rows_h4096<<<dim3(rows,4,1),dim3(256,1,1)>>>(
        input,tokens,experts,gate_tab,up_tab,fixed_gate,fixed_up,kPreWidth,rows);
    check(cudaGetLastError(),"pre reference");
}
void quant_reference(const __nv_bfloat16* input,unsigned char* fp8,float* scale,
                     unsigned rows,unsigned width) {
    per_token_group_quant_fp8<<<dim3(rows,width/128,1),dim3(128,1,1)>>>(input,fp8,scale,rows,width);
    check(cudaGetLastError(),"quant reference");
}
void pre_candidate(const __nv_bfloat16* input,const int* tokens,const int* experts,
                   const unsigned long long* gate_tab,const unsigned long long* up_tab,
                   unsigned char* gate,float* gate_scale,unsigned char* up,float* up_scale,
                   unsigned width,unsigned rows,dim3 grid,dim3 block) {
    exl3_w2a8_h128_pre_dual_emit_h4096<<<grid,block>>>(
        input,tokens,experts,gate_tab,up_tab,gate,gate_scale,up,up_scale,width,rows);
    check(cudaGetLastError(),"pre candidate");
}
void post_reference(__nv_bfloat16* gate,const __nv_bfloat16* up,const int* experts,
                    const unsigned long long* gate_tab,const unsigned long long* up_tab,
                    const unsigned long long* down_tab,unsigned rows) {
    exl3_h128_post_silu_pre_rows<<<dim3(rows,2,1),dim3(256,1,1)>>>(
        gate,up,experts,gate_tab,up_tab,down_tab,kPostWidth);
    check(cudaGetLastError(),"post reference");
}
void post_candidate(const __nv_bfloat16* gate,const __nv_bfloat16* up,const int* experts,
                    const unsigned long long* gate_tab,const unsigned long long* up_tab,
                    const unsigned long long* down_tab,unsigned char* output,float* scale,
                    unsigned width,unsigned rows,dim3 grid,dim3 block) {
    exl3_w2a8_h128_post_silu_pre_emit_h2048<<<grid,block>>>(
        gate,up,experts,gate_tab,up_tab,down_tab,output,scale,width,rows);
    check(cudaGetLastError(),"post candidate");
}
// END emitter probe kernel contract

std::size_t byte_mismatches(const Buffer& actual,const Buffer& expected,std::size_t bytes) {
    const auto a=actual.download(), e=expected.download();
    std::size_t count=0; for(std::size_t i=0;i<bytes;++i) count+=a[i]!=e[i]; return count;
}

struct ParityCounts {
    std::size_t pre_gate_fp8_mismatches=0;
    std::size_t pre_gate_scale_mismatches=0;
    std::size_t pre_up_fp8_mismatches=0;
    std::size_t pre_up_scale_mismatches=0;
    std::size_t post_down_fp8_mismatches=0;
    std::size_t post_down_scale_mismatches=0;
};
// BEGIN emitter byte parity contract
void compare_pre(ParityCounts& count,const Buffer& gate,const Buffer& gate_expected,
                 const Buffer& gate_scale,const Buffer& gate_scale_expected,const Buffer& up,
                 const Buffer& up_expected,const Buffer& up_scale,const Buffer& up_scale_expected,
                 unsigned rows) {
    count.pre_gate_fp8_mismatches+=byte_mismatches(gate,gate_expected,rows*kPreWidth);
    count.pre_gate_scale_mismatches+=byte_mismatches(gate_scale,gate_scale_expected,rows*(kPreWidth/128)*sizeof(float));
    count.pre_up_fp8_mismatches+=byte_mismatches(up,up_expected,rows*kPreWidth);
    count.pre_up_scale_mismatches+=byte_mismatches(up_scale,up_scale_expected,rows*(kPreWidth/128)*sizeof(float));
}
void compare_post(ParityCounts& count,const Buffer& output,const Buffer& expected,
                  const Buffer& scale,const Buffer& scale_expected,unsigned rows) {
    count.post_down_fp8_mismatches+=byte_mismatches(output,expected,rows*kPostWidth);
    count.post_down_scale_mismatches+=byte_mismatches(scale,scale_expected,rows*(kPostWidth/128)*sizeof(float));
}
// END emitter byte parity contract

// BEGIN emitter poison contract
constexpr unsigned char poison_a=0xa5, poison_b=0x5a;
// Each poison_independent run verifies prefix_guard, suffix_guard, inactive_rows,
// active_fp8, active_scales, and the materialized legacy_bf16 oracle. Fixtures
// contain an all_zero_group, signed_zero, and bf16_boundary values.
void verify_poison(const Buffer& output,std::size_t active,unsigned char poison,const char* region) {
    const auto bytes=output.download();
    for(std::size_t i=active;i<bytes.size();++i) if(bytes[i]!=poison) fail(region);
    if(!output.guards_clean()) fail("output-guard");
}
template <typename T> void verify_unchanged_tail(const Buffer& output,
                                                 const std::vector<T>& original,
                                                 std::size_t active) {
    const auto bytes = output.download();
    const std::size_t offset = active * sizeof(T);
    if (std::memcmp(bytes.data() + offset,
                    reinterpret_cast<const unsigned char*>(original.data()) + offset,
                    bytes.size() - offset) != 0) fail("legacy-tail-changed");
    if (!output.guards_clean()) fail("legacy-output-guard");
}
// END emitter poison contract

struct Geometry { const char* name; unsigned width,rows; dim3 grid,block; };

// BEGIN emitter malformed geometry contract
const Geometry bad_pre[]={
 {"pre-block-x",kPreWidth,7,dim3(7,4,1),dim3(128,1,1)},
 {"pre-block-y",kPreWidth,7,dim3(7,4,1),dim3(256,2,1)},
 {"pre-block-z",kPreWidth,7,dim3(7,4,1),dim3(256,1,2)},
 {"pre-grid-x",kPreWidth,7,dim3(6,4,1),dim3(256,1,1)},
 {"pre-grid-y",kPreWidth,7,dim3(7,3,1),dim3(256,1,1)},
 {"pre-grid-z",kPreWidth,7,dim3(7,4,2),dim3(256,1,1)},
 {"pre-rows",kPreWidth,8,dim3(7,4,1),dim3(256,1,1)},
 {"pre-k",4095,7,dim3(7,4,1),dim3(256,1,1)}};
const Geometry bad_post[]={
 {"post-block-x",kPostWidth,7,dim3(7,2,1),dim3(128,1,1)},
 {"post-block-y",kPostWidth,7,dim3(7,2,1),dim3(256,2,1)},
 {"post-block-z",kPostWidth,7,dim3(7,2,1),dim3(256,1,2)},
 {"post-grid-x",kPostWidth,7,dim3(6,2,1),dim3(256,1,1)},
 {"post-grid-y",kPostWidth,7,dim3(7,1,1),dim3(256,1,1)},
 {"post-grid-z",kPostWidth,7,dim3(7,2,2),dim3(256,1,1)},
 {"post-rows",kPostWidth,8,dim3(7,2,1),dim3(256,1,1)},
 {"post-n",2047,7,dim3(7,2,1),dim3(256,1,1)}};
void guard_sync(const char* name) {
    check(cudaGetLastError(),name); check(cudaDeviceSynchronize(),name); // canary=clean
}
// The invalid-shape launch passes nullptr inputs, proving guards precede dereference.
// END emitter malformed geometry contract

struct PreFixture {
    std::vector<std::uint16_t> input_h=make_values(kTokens,kPreWidth,11);
    std::vector<std::int32_t> token_h=sorted_token_ids(), expert_h=sorted_expert_ids();
    std::vector<std::uint16_t> gate_sign_h=make_signs(kPreWidth,17), up_sign_h=make_signs(kPreWidth,43);
    Buffer input{input_h.size()*2},tokens{token_h.size()*4},experts{expert_h.size()*4};
    Buffer gate_sign{gate_sign_h.size()*2},up_sign{up_sign_h.size()*2},gate_tab{kExperts*8},up_tab{kExperts*8};
    PreFixture(){input.upload(input_h);tokens.upload(token_h);experts.upload(expert_h);
      gate_sign.upload(gate_sign_h);up_sign.upload(up_sign_h);
      gate_tab.upload(make_table(gate_sign,kPreWidth));up_tab.upload(make_table(up_sign,kPreWidth));}
};

void run_pre(ParityCounts& counts,std::uint64_t& input_hash,std::uint64_t& routing_hash,std::uint64_t& tables_hash) {
    PreFixture f; input_hash=hash_vector(input_hash,f.input_h);
    routing_hash=hash_vector(hash_vector(routing_hash,f.token_h),f.expert_h);
    tables_hash=hash_vector(hash_vector(tables_hash,f.gate_sign_h),f.up_sign_h);
    const std::size_t fp8_n=static_cast<std::size_t>(kMaxRows)*kPreWidth;
    const std::size_t scale_n=static_cast<std::size_t>(kMaxRows)*(kPreWidth/128)*sizeof(float);
    Buffer gate_bf16(fp8_n*2),up_bf16(fp8_n*2),fixed_gate(fp8_n*2),fixed_up(fp8_n*2);
    Buffer eg(fp8_n),esg(scale_n),eu(fp8_n),esu(scale_n),ag(fp8_n),asg(scale_n),au(fp8_n),asu(scale_n);
    for(unsigned rows:row_cases) for(unsigned routing=0;routing<2;++routing) {
      const int* token_ptr=routing?identity_token_ids:reinterpret_cast<int*>(f.tokens.data);
      gate_bf16.fill(poison_a);up_bf16.fill(poison_a);fixed_gate.fill(poison_a);fixed_up.fill(poison_a);
      eg.fill(poison_a);esg.fill(poison_a);eu.fill(poison_a);esu.fill(poison_a);
      pre_reference(reinterpret_cast<__nv_bfloat16*>(f.input.data),token_ptr,reinterpret_cast<int*>(f.experts.data),
       reinterpret_cast<unsigned long long*>(f.gate_tab.data),reinterpret_cast<unsigned long long*>(f.up_tab.data),
       reinterpret_cast<__nv_bfloat16*>(gate_bf16.data),reinterpret_cast<__nv_bfloat16*>(up_bf16.data),
       reinterpret_cast<__nv_bfloat16*>(fixed_gate.data),reinterpret_cast<__nv_bfloat16*>(fixed_up.data),rows);
      quant_reference(reinterpret_cast<__nv_bfloat16*>(gate_bf16.data),eg.data,reinterpret_cast<float*>(esg.data),rows,kPreWidth);
      quant_reference(reinterpret_cast<__nv_bfloat16*>(up_bf16.data),eu.data,reinterpret_cast<float*>(esu.data),rows,kPreWidth);
      for(unsigned char poison:{poison_a,poison_b}) {ag.fill(poison);asg.fill(poison);au.fill(poison);asu.fill(poison);
       pre_candidate(reinterpret_cast<__nv_bfloat16*>(f.input.data),token_ptr,reinterpret_cast<int*>(f.experts.data),
        reinterpret_cast<unsigned long long*>(f.gate_tab.data),reinterpret_cast<unsigned long long*>(f.up_tab.data),
        ag.data,reinterpret_cast<float*>(asg.data),au.data,reinterpret_cast<float*>(asu.data),kPreWidth,rows,dim3(rows,4,1),dim3(256,1,1));
       check(cudaDeviceSynchronize(),"pre parity"); compare_pre(counts,ag,eg,asg,esg,au,eu,asu,esu,rows);
       if(byte_mismatches(ag,au,rows*kPreWidth)==0)fail("gate-up-not-distinct");
       verify_poison(ag,rows*kPreWidth,poison,"pre-tail");verify_poison(asg,rows*(kPreWidth/128)*sizeof(float),poison,"pre-scale-tail");
       verify_poison(au,rows*kPreWidth,poison,"up-tail");verify_poison(asu,rows*(kPreWidth/128)*sizeof(float),poison,"up-scale-tail");}
      if(byte_mismatches(gate_bf16,fixed_gate,rows*kPreWidth*2)||byte_mismatches(up_bf16,fixed_up,rows*kPreWidth*2))fail("fixed-pre");
      verify_poison(eg,rows*kPreWidth,poison_a,"reference-gate-tail");verify_poison(esg,rows*(kPreWidth/128)*sizeof(float),poison_a,"reference-gate-scale-tail");
      verify_poison(eu,rows*kPreWidth,poison_a,"reference-up-tail");verify_poison(esu,rows*(kPreWidth/128)*sizeof(float),poison_a,"reference-up-scale-tail");
      verify_poison(gate_bf16,rows*kPreWidth*2,poison_a,"legacy-pre-tail");
      verify_poison(up_bf16,rows*kPreWidth*2,poison_a,"legacy-up-tail");
      verify_poison(fixed_gate,rows*kPreWidth*2,poison_a,"fixed-gate-tail");
      verify_poison(fixed_up,rows*kPreWidth*2,poison_a,"fixed-up-tail");
    }
    for(const Geometry& bad:bad_pre){ag.fill(poison_a);asg.fill(poison_a);au.fill(poison_a);asu.fill(poison_a);
      const bool null_inputs=std::strcmp(bad.name,"pre-k")==0;
      pre_candidate(null_inputs?nullptr:reinterpret_cast<__nv_bfloat16*>(f.input.data),
       null_inputs?nullptr:reinterpret_cast<int*>(f.tokens.data),null_inputs?nullptr:reinterpret_cast<int*>(f.experts.data),
       null_inputs?nullptr:reinterpret_cast<unsigned long long*>(f.gate_tab.data),null_inputs?nullptr:reinterpret_cast<unsigned long long*>(f.up_tab.data),
       ag.data,reinterpret_cast<float*>(asg.data),au.data,reinterpret_cast<float*>(asu.data),bad.width,bad.rows,bad.grid,bad.block);
      guard_sync(bad.name);verify_poison(ag,0,poison_a,bad.name);verify_poison(asg,0,poison_a,bad.name);
      verify_poison(au,0,poison_a,bad.name);verify_poison(asu,0,poison_a,bad.name);}
}

void run_post(ParityCounts& counts,std::uint64_t& input_hash,std::uint64_t& routing_hash,std::uint64_t& tables_hash) {
    auto gate_h=make_values(kMaxRows,kPostWidth,71),up_h=make_values(kMaxRows,kPostWidth,97);auto expert_h=sorted_expert_ids();
    auto gs_h=make_signs(kPostWidth,5),us_h=make_signs(kPostWidth,23),ds_h=make_signs(kPostWidth,61);
    input_hash=hash_vector(hash_vector(input_hash,gate_h),up_h);routing_hash=hash_vector(routing_hash,expert_h);
    tables_hash=hash_vector(hash_vector(hash_vector(tables_hash,gs_h),us_h),ds_h);
    Buffer gate(gate_h.size()*2),oracle(gate_h.size()*2),up(up_h.size()*2),experts(expert_h.size()*4);
    Buffer gs(gs_h.size()*2),us(us_h.size()*2),ds(ds_h.size()*2),gt(kExperts*8),ut(kExperts*8),dt(kExperts*8);
    gate.upload(gate_h);up.upload(up_h);experts.upload(expert_h);gs.upload(gs_h);us.upload(us_h);ds.upload(ds_h);
    gt.upload(make_table(gs,kPostWidth));ut.upload(make_table(us,kPostWidth));dt.upload(make_table(ds,kPostWidth));
    const std::size_t fp8_n=static_cast<std::size_t>(kMaxRows)*kPostWidth;
    const std::size_t scale_n=static_cast<std::size_t>(kMaxRows)*(kPostWidth/128)*sizeof(float);
    Buffer expected(fp8_n),expected_scale(scale_n),actual(fp8_n),actual_scale(scale_n);
    for(unsigned rows:row_cases){oracle.upload(gate_h);
      expected.fill(poison_a);expected_scale.fill(poison_a);
      post_reference(reinterpret_cast<__nv_bfloat16*>(oracle.data),reinterpret_cast<__nv_bfloat16*>(up.data),reinterpret_cast<int*>(experts.data),
       reinterpret_cast<unsigned long long*>(gt.data),reinterpret_cast<unsigned long long*>(ut.data),reinterpret_cast<unsigned long long*>(dt.data),rows);
      quant_reference(reinterpret_cast<__nv_bfloat16*>(oracle.data),expected.data,reinterpret_cast<float*>(expected_scale.data),rows,kPostWidth);
      for(unsigned char poison:{poison_a,poison_b}){actual.fill(poison);actual_scale.fill(poison);
       post_candidate(reinterpret_cast<__nv_bfloat16*>(gate.data),reinterpret_cast<__nv_bfloat16*>(up.data),reinterpret_cast<int*>(experts.data),
        reinterpret_cast<unsigned long long*>(gt.data),reinterpret_cast<unsigned long long*>(ut.data),reinterpret_cast<unsigned long long*>(dt.data),
        actual.data,reinterpret_cast<float*>(actual_scale.data),kPostWidth,rows,dim3(rows,2,1),dim3(256,1,1));
       check(cudaDeviceSynchronize(),"post parity");compare_post(counts,actual,expected,actual_scale,expected_scale,rows);
       verify_poison(actual,rows*kPostWidth,poison,"post-tail");verify_poison(actual_scale,rows*(kPostWidth/128)*sizeof(float),poison,"post-scale-tail");}
      verify_poison(expected,rows*kPostWidth,poison_a,"reference-post-tail");
      verify_poison(expected_scale,rows*(kPostWidth/128)*sizeof(float),poison_a,"reference-post-scale-tail");
      verify_unchanged_tail(oracle,gate_h,rows*kPostWidth);}
    for(const Geometry& bad:bad_post){actual.fill(poison_a);actual_scale.fill(poison_a);const bool null_inputs=std::strcmp(bad.name,"post-n")==0;
      post_candidate(null_inputs?nullptr:reinterpret_cast<__nv_bfloat16*>(gate.data),null_inputs?nullptr:reinterpret_cast<__nv_bfloat16*>(up.data),
       null_inputs?nullptr:reinterpret_cast<int*>(experts.data),null_inputs?nullptr:reinterpret_cast<unsigned long long*>(gt.data),
       null_inputs?nullptr:reinterpret_cast<unsigned long long*>(ut.data),null_inputs?nullptr:reinterpret_cast<unsigned long long*>(dt.data),
       actual.data,reinterpret_cast<float*>(actual_scale.data),bad.width,bad.rows,bad.grid,bad.block);
      guard_sync(bad.name);verify_poison(actual,0,poison_a,bad.name);verify_poison(actual_scale,0,poison_a,bad.name);}
}

bool valid_build_id(const char* id){if(std::strlen(id)!=64)return false;for(unsigned i=0;i<64;++i)
 if(!((id[i]>='0'&&id[i]<='9')||(id[i]>='a'&&id[i]<='f')))return false;return true;}
}  // namespace probe

// BEGIN emitter bounded output contract
int main(int argc,char** argv){
 if(argc!=1){std::fprintf(stderr,"usage: %s\n",argv[0]);return 2;}
 const char* build_id = W2A8_EMIT_PROBE_BUILD_ID;if(!probe::valid_build_id(build_id)){std::fprintf(stderr,"FAIL invalid build id\n");return 2;}
 probe::ParityCounts counts;std::uint64_t input_hash = 0xcbf29ce484222325ull,routing_hash = input_hash,tables_hash = input_hash;
 probe::run_pre(counts,input_hash,routing_hash,tables_hash);probe::run_post(counts,input_hash,routing_hash,tables_hash);
 const std::size_t mismatches = counts.pre_gate_fp8_mismatches+counts.pre_gate_scale_mismatches+counts.pre_up_fp8_mismatches+
  counts.pre_up_scale_mismatches+counts.post_down_fp8_mismatches+counts.post_down_scale_mismatches;if(mismatches)probe::fail("byte-mismatch");
 int driver = 0,runtime = 0;probe::check(cudaDriverGetVersion(&driver),"driver");probe::check(cudaRuntimeGetVersion(&runtime),"runtime");
 cudaDeviceProp prop{};probe::check(cudaGetDeviceProperties(&prop,0),"device");
 std::printf("build_id=%s\n",build_id);std::printf("device_uuid=");for(unsigned char byte:prop.uuid.bytes)std::printf("%02x",byte);
 std::printf(" driver=%d runtime=%d\n",driver,runtime);
 std::printf("input_hash=%016llx routing_hash=%016llx tables_hash=%016llx\n",(unsigned long long)input_hash,(unsigned long long)routing_hash,(unsigned long long)tables_hash);
 std::printf("pre_cases=16 post_cases=8 geometry_cases=16 mismatches=0 guards=clean\n");std::printf("result=PASS\n");return 0;
}
// END emitter bounded output contract
