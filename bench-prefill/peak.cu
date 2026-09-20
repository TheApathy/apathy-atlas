// cuBLAS tensor-core peak probe on this GB10: BF16 (f32 acc) and FP8 e4m3 GEMMs at prefill shapes.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <vector>
#include <chrono>
#define CK(x) do{auto e=(x); if(e){printf("err %d at %d\n",(int)e,__LINE__);return 1;}}while(0)
int main(){
  int shapes[][3]={{2048,16384,5120},{2048,12288,5120},{2048,5120,6144},{2048,34816,5120},{2048,5120,17408}};
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  void* ws; cudaMalloc(&ws, 64<<20);
  for (auto& s: shapes){
    int M=s[0],N=s[1],K=s[2];
    for (int mode=0; mode<2; mode++){
      cudaDataType at = mode==0?CUDA_R_16BF:CUDA_R_8F_E4M3;
      void *A,*B,*C; cudaMalloc(&A,(size_t)M*K*2); cudaMalloc(&B,(size_t)N*K*2); cudaMalloc(&C,(size_t)M*N*2);
      cudaMemset(A,0,(size_t)M*K*2); cudaMemset(B,0,(size_t)N*K*2);
      cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
      cublasOperation_t tA=CUBLAS_OP_T, tB=CUBLAS_OP_N;
      CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
      CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
      // D[N,M] col-major = W[N,K]^T-ish: compute C(N x M) = W(K x N)^T * X(K x M)
      cublasLtMatrixLayout_t la,lb,lc;
      CK(cublasLtMatrixLayoutCreate(&la, at, K, N, K));
      CK(cublasLtMatrixLayoutCreate(&lb, at, K, M, K));
      CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
      cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
      size_t wss=64<<20; CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wss,sizeof(wss)));
      cublasLtMatmulHeuristicResult_t heur[8]; int nres=0;
      CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, heur, &nres));
      if(nres==0){printf("no algo M=%d N=%d K=%d mode=%d\n",M,N,K,mode);continue;}
      float alpha=1,beta=0;
      double best=1e9;
      for(int a=0;a<nres && a<4;a++){
        for(int w=0;w<3;w++) CK(cublasLtMatmul(lt,op,&alpha,A,la,B,lb,&beta,C,lc,C,lc,&heur[a].algo,ws,wss,0));
        cudaDeviceSynchronize();
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for(int it=0;it<10;it++) CK(cublasLtMatmul(lt,op,&alpha,A,la,B,lb,&beta,C,lc,C,lc,&heur[a].algo,ws,wss,0));
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=10;
        if(ms<best) best=ms;
      }
      double tf=2.0*M*N*K/best/1e9;
      printf("%s M=%d N=%d K=%d  %.3f ms  %.1f TFLOP/s\n", mode==0?"BF16":"FP8 ", M,N,K,best,tf);
      cudaFree(A);cudaFree(B);cudaFree(C);
    }
  }
  return 0;
}
