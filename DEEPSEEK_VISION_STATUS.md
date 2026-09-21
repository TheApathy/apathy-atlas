# DeepSeek Flash Vision bring-up and prefill census

## 2026-09-08: current-tree Vision restored; native DSpark and prefill qualification

The latest exact same-checkpoint result is a **914.8955 effective prefill tok/s**
median at exactly 2410 prompt tokens (2634.181 ms median server TTFT), with
**26.6968 decode tok/s** over the short 32-token response. This is a fresh
strict HC-off R11 run using the immutable production wrapper; the warmup and
all five measured runs preserve the reference output SHA256
`7a2bf862af094a3b2cb670329d6072b961805a37b578659ddd87a4cb3cf0871a`.
It is 0.37% below the earlier R8 result, so the wrapper recipe is reproduced on
R11 but does not establish a new speed win. Both remain well below the 2000
tok/s objective.

| Current arm | Prompt/response shape | Effective prefill | API decode | Qualification boundary |
|---|---:|---:|---:|---|
| General DSpark + fused post |2048 /32|746.12 tok/s|25.7431 tok/s on the separate exact142-token passage|Six basic text/synthetic-PNG/order/repeat checks and exact passage pass|
| Strict max, HC off (R11 reproduction)|2410 /32|914.8955 tok/s|26.6968 tok/s|Exact shape only; W2A8 changes activation precision; warmup +5 outputs exact|
| Strict max, HC on (rejected)|2410 /32|908.7784 tok/s|26.8791 tok/s|Exact output, but 1.03% slower end-to-end than HC off|

The strict arm uses W2A8, fused GU, N256 down, fused unpermute, KV alias and
inverse RoPE with `ATLAS_PREFILL_MAX_REQUIRE_ARMS=1`. It intentionally fails
closed outside the exact 2410-by-topk6 routed-row shape. The earlier measured
918.2851 result used the immutable R8 ELF (SHA256
`35d92df8809b68e74d7f0e9142f567031031b73446e6b8de5391a575fa9286eb`).
The fresh reproduction uses the wrapper's immutable R11 ELF (SHA256
`354d92ea09496ef6f3be41025ff4daa2a0f2ac35cda27fd5509fb74e267d8722`).
R11 changes only the optional HC/RMS planner and the winning wrapper keeps that
arm off. The R11 raw responses and log are under
`/var/tmp/atlas-deepseek-vision-spec.EKOuMy1l/dspark-max2410-live.9D8RX9H2/`.

The 2000 tok/s target remains unachieved and does not yet have a qualified
single smoking gun. The startup MoE-transpose capacity warning is generic and
does not govern the EXL3 trellis path used by this checkpoint, so it must not be
used to explain the measured 914.8955 result. Existing exact-shape attention
and HC candidates either miss their registered component threshold or lose
end to end; a fresh disjoint on-GPU profile of the strict path is required
before selecting the next implementation.

The R11 HC/RMS path now admits the Vision checkpoint's finite positive
`rms_norm_eps=1e-20` only when the runtime HC norm epsilon matches the config.
Its production-shape GPU oracle is byte-exact with clean poison/canaries and
shows a 1.2157x isolated kernel speedup, but the full model rejected it because
end-to-end TTFT regressed. This keeps an isolated kernel win from being promoted
as a model win.

R5 closes a same-binary native DSpark/plain comparison on the retained Vision checkpoint. Both arms passed all six text/PNG/order/repeat-state checks and six exact142-token passages. Median of five measured passage runs (one warmup excluded):

| R5 mode | Exact-passage decode | Effective prefill256 | Effective prefill2048 | Basic text/image checks |
|---|---:|---:|---:|---:|
| Plain |18.1159tok/s|360.2058tok/s|731.7277tok/s|6/6|
| Native DSpark, adaptive |20.7802tok/s|358.8058tok/s|721.8505tok/s|6/6|

The short-copy decode gain is14.7073% in this single ordered same-binary comparison, not a coding-performance result or universal speedup. Prefill is prompt tokens divided by server TTFT, not isolated GPU time. TTFT at2048: plain2798.855ms, DSpark configuration2837.152ms. All50 raw records passed hash/count/HTTP200/curl0/zero-cache/finite-timing checks; all25 requests match and24/25 outputs match. The sole difference is the2048-token warmup summary wording: both are coherent, but **greedy-output parity is not established**. At long context the adaptive throughput gate preferred plain decoding. No8192 bin, Weschera coding or broad natural-image/OCR/JPEG qualification is claimed.

Current-tree fixes include text-only Vision preparation (zero images must be valid and clear pending state), architecture-scoped native compressed-YaRN image admission within the original context window, rejection of unsupported multi-choice image requests, and balanced rejection metrics. Both live admission failures and their binary/log/response evidence are retained.

Embedded native DSpark is present in the retained `DeepSeek-V4-Flash-Vision-EXL3-K2-c171bea5` checkpoint. R3 loader confirmed **9316 tensors /5455344156B shared zero-copy** after fixing `--model-from-path` identity resolution. DSpark base RoPE is now kept separate from target compressed RoPE, and bootstrap/short-chain verification uses the full capture/compressor-aware target path. Qualification remains explicit/default-off, same-checkpoint only, gamma6/capture1/masked verification/C1/no-prefix/no-HSS/no-adapters, with unsafe experimental modes rejected. This is **native DSpark, not a genuine DFlash2 drafter**.

R3 speculative startup stopped at a new factory guard that misclassified the shared speculation switch. R4 fixes that classification and installs all three DSpark stages, but proposals failed the native FP8 shared-expert wide-verify guard. Its six correct smoke outputs were plain fallback, **not working DSpark**. R5 fixes native FP8 wide verification using unchanged scalar GEMV/clamp10 math per row with checked scratch geometry/capacities/alignment. **R5 native DSpark now passes6/6 text/PNG/order/repeat-state checks and6/6 exact142-token passages, with20.7802 decode tok/s median.** Actual five-token proposals, BATCHEDn6 verification and20 full-accept plus161 partial-accept counters prove real execution; no proposal/runtime errors. This is a short passage-copy result, not coding or broad-vision qualification.

At2078tokens Atlas's throughput gate switched to plain decode because measured DSpark was slower (15.3 vs17.4tok/s); longer-context decode is adaptive, not purely speculative. Both R5 servers are drained; GPU/8977 were empty after the identity-checked shutdowns. All prior artifacts remain preserved.

Evidence and exact runtime identities: `/var/tmp/atlas-deepseek-vision-spec.EKOuMy1l/REPORT.md`, `comparison-r5.json`, `{plain,dspark}-r5-review.json`, and `{plain,dspark}-r5-{smoke,passage,census}/`. Comparison SHA256 `6fee83cc6ad558acdfdbf5729921df7b95d228a1986c20dd09aa2d856cd24d8d`. Native ELF SHA256 `2fa733162881237b5782a221a66bfbcdc889686a51b250f3994bc6eb21a17ca6`. Recorded server PIDs have exited; never reuse their provenance as a live binding.

### Reproduce the current local recipes on Reaper

From this checkout, run `bash bench/deepseek-v4/serve_vision_qualified.sh dspark`
(or `plain`) for the general BF16/N128 recipe. Run it with
`dspark-max2410` only for the strict 2410-token W2A8 qualification shape. This
is an explicit experimental localhost8977/C1/12288-context recipe, **not a
default-service or production promotion**. It refuses a busy GPU/port, checks
the frozen R11 build-source manifest and binds fresh process provenance to the
pinned R11 ELF/checkpoint config. It prints a new run directory containing
`server.log` and `provenance.json`, so repeated launches do not overwrite
benchmark evidence. Model startup takes roughly9–10minutes. The script depends
on preserved local R11 artifacts and the coordination helper; it does not
download/build or stop other services. The max recipe has now reproduced the
measured R8 winner on R11 within 0.37%.

The general R11 DSpark arm now passes broader natural-image JPEG, native JPEG,
and OCR PNG probes. The converted natural aurora JPEG returned `Aurora`, the
native NVIDIA background JPEG returned `Nvidia`, and the OCR fixture returned
the exact `Select a File|foo.sh`. The original WebP was correctly rejected with
HTTP 400, so WebP is still an unsupported admission gap rather than a vision
success. Artifacts are in
`/var/tmp/atlas-deepseek-vision-spec.EKOuMy1l/broad-vision-r11.4474yk03/`;
summary SHA256 is
`e0f87122e23521e4af13aef337a1c05076e61a0f48e4e86f05abb1fdc2f835a7`.

The canonical five-run Weschera MinHeap speed result is **25.7287 tok/s
median** for 400 output tokens, with an identical output SHA across all runs.
Because 400 tokens ends mid-method, it is a deterministic speed result only.
A separate 1361-token natural completion parsed, executed, and passed 100
randomized heapify cases plus 300 mixed operations in each of 100 trials
(30,000 mixed operations). Its receipt SHA256 is
`7449e41ba2c26d32bb4a56cbd9f5ed85818bba46d34e658de60bc411f993f221`.

A genuine external Vision DFlash2 drafter remains unavailable. The only local
target-component bundle differs by value hash at `embed.weight` from the
current Vision checkpoint, and no compatible hidden-state corpus or trained
drafter checkpoint exists. It must not be relabelled or attached. The working
speculative path is the checkpoint's embedded native DSpark through Atlas's
DFlash scheduler, **not genuine external DFlash2**.

Production remains unqualified for WebP, encoder numerical parity,
long-context/request-state coverage, genuine external DFlash2, and 2000+
prefill. The general BF16 arm has the stronger quality boundary; the strict
W2A8 arm must not inherit broad quality from output equality on one prompt.
The next prefill implementation must be selected from disjoint on-GPU timing of
the actual EXL3 strict path, not from the inapplicable generic transpose warning.

## Historical 2026-09-05 snapshot

Snapshot: 2026-09-05 10:35:03 UTC. Actual DeepSeek Flash Vision now measures **731.17 effective prefill tok/s at2048 tokens** (median TTFT2.801s) and361.72 at256 (707.73ms), with18.0184tok/s exact-passage decode. Existing BF16 P2 N128/K64 preserves all25 tested outputs and improves prefill3.506%/4.489% versus N64. Reverse P1 control reproduces the original slow baseline within1%. Basic vision6/6 passes; broader vision, encoder/JPEG parity, DFlash and2000+prefill remain unqualified. Both control/candidate servers drained;108GiB available/noGPU/listener. Source and binary unchanged; no global default promotion.

## V28/V29: reverse control confirms the gain; N128 improves prefill further

All runs use the same actual Vision checkpoint, frozen V25 native binary09526881... and source3e7f89b5.... V28 restored the exact V26 four-environment P1 recipe. Its twelve warm/measured request/output hashes, token counts, cache0, finish reasons and exact canary match V26. Five-run median prefill returned to31.1713/163.2369tok/s at256/2048, within1% of the original30.9201/161.9648. The large direct-P2 win therefore survives a reverse-order control.

V29 changes ONLY `ATLAS_EXL3_PREFILL_N128=0 -> 1` relative to V27. All other flags, argv, CWD, binary/config/tokenizer/template identities match. The existing production-shape P1/P2 and N64/N128 parity plus width memcheck gates remain valid because source and all207PTX are unchanged. Six-case vision smoke passed first, then the unchanged frozen census and exact142-token passage-copy drivers ran serially, each with one warmup and five measured samples.

| Metric | V28 reverse P1 | V27 P2 N64 | V29 P2 N128 |
|---|---:|---:|---:|
|256-token effective prefill|31.1713 tok/s|346.1797 tok/s|361.7199 tok/s|
|256-token server TTFT|8212.670ms|739.500ms|707.730ms|
|2048-token effective prefill|163.2369 tok/s|706.4023 tok/s|731.1717 tok/s|
|2048-token server TTFT|12546.181ms|2899.198ms|2800.984ms|
|142-token passage decode|not run|18.1565 tok/s|18.0184 tok/s|
|165-token passage TTFT|not run|684.401ms|653.911ms|

N128 prefill ranges:256=360.6739-362.6752tok/s,2048=729.9790-733.6334. Decode range17.9196-18.1505tok/s is effectively unchanged; no significant decode/coding/DFlash improvement claimed. Prefill is prompt/serverTTFT, not isolated GPU throughput. N128 is a measured candidate, not a globally best tile or default; repeated/order-reversed N64/N128 comparison remains appropriate before promotion.

All25 V29 outputs and all census/passage request hashes, counts/cache/finishes equal V27. Root independently reclosed38 raw records:13 V28 plus25 V29, including SHA256 of every request/response/output, HTTP200/curl0, explicit cache0/count sums, exact content/finish and finite positive timings. The existing4.5e-16 relative JSON TTFT serialization allowance is unchanged. Source, native binary, three driver binaries, config/tokenizer/template rehashed unchanged. No CUDA/OOM/nonfinite/watchdog fault or competing workload.

V29 individual smoke TTFTs: text452.30ms initially/239.91ms afterimages; singleimage655.62-659.92ms; two-image996.41ms. These are individual basic PNG color/order/repeatability samples, not image-performance medians or broader multimodal qualification. V7 encoder numerical failure and JPEG/OCR/natural-image gates remain open.

Full immutable artifact paths and hashes: `/var/tmp/atlas-deepseek-vision-v28-v29-root-review.json`. V28 census summary `/var/tmp/atlas-prefill-census-deepseek-vision-v28-p1-control/summary.json` SHA9dc0d0fe...; V29 census `/var/tmp/atlas-prefill-census-deepseek-vision-v29-n128/summary.json` SHAe8d33abd...; smoke `/var/tmp/atlas-deepseek-vision-v29-n128-smoke/summary.json` SHAb981efe0...; passage `/var/tmp/atlas-deepseek-passage-decode-v29-n128/summary.json` SHA7386e2c9.... V29 provenance `/var/tmp/atlas-deepseek-vision-v29-n128-provenance.json` SHAf37a0448... contains the exact reproducible recipe; its recorded PID is exited and MUST be rebound, never reused as live provenance.

V28 PID1709484/start4691117 and V29 PID1726653/start4756904 both drained with SIGINT/nativeexit130. V29 startup10:17:05 -> ready10:25:26 is separate from requestTTFT. Postflight before10:30:38 release claim:108GiB available, no GPU process/listener/compiler. V29 logSHA4b64952b.... No production source changed in V26-V29.

### Reuse findings and next gate

ModelForge's existing header inventory is useful evidence but currently labels this actual Vision EXL3 checkpoint non-MoE/expert0/quantization-null. Do not use that report as model admission or a numerical/performance oracle; no ModelForge edits were made.

Reuse Atlas's existing `ATLAS_PROFILE` attention/MoE/layer-glue hooks for the next fresh root-owned profiling window on the frozen N128 recipe. Retain the exact known2048-token request/output and provenance; profiling-only timings receive no performance credit. `scripts/prefill_waterfall.py` can parse buckets, but naively totals nested wrappers/components and its optional timestamp argument compares seconds-of-day despite documenting epoch. Isolate one request and sum disjoint buckets. `ATLAS_PROFILE_FIRST` would be consumed by a canary, so do not accidentally profile only that tiny prompt.

Historical `docs/PREFILL-CAMPAIGN-2026-08-10.md` found no end-to-end gain from `ATLAS_V4_STAGE_SYNCS=0` on the older text model. That is counterevidence against blindly chasing synchronization removal, not a current Vision measurement. Existing Nsight wrappers hard-code an old text checkpoint/worktree/recipe and cannot run unchanged. Profile actual dispatched kernels before selecting the next candidate; preserve existing quantization, semantic gates, and all vision limitations.

## V27: direct BF16 prefill candidate passes GPU and full-model gates

The direct path consumes decoded BF16 weight fragments without materializing P1's full expert-weight scratch and avoids the per-layer expert-offset host copy. BF16 activations/weights, K16 MMA accumulation order, H128 rotations, clamp10, Vision routing, native sharedFP8 and attention settings remain unchanged. Root and independent read-only agent audited the existing path. No new kernel/production source was required.

Fresh locked/offline native probe build `/var/tmp/atlas-deepseek-vision-v27-probes` took42.15s,207modules/35overrides; ALL207PTX byte-identical to frozenV25. Existing `exl3_gemv_microtest` passes all gates including exact P1 H128/dequant, K2/K3 P2-vs-P1 byteparity at1/63/64/65/129expertrows, optional fusedpost/tail and M1..16 replay. LogSHA `42870c85aedab6173c9e86363e098133adf7a4fbe456ae5f331f88e9ba83a708`.

Existing `exl3_prefill_n128_microtest` passes both production matrix shapes: generic/fixedN64/N128/N256 outputs exact and wrongblock guards untouched. Directional GU/down times: N64=.364/.258ms,N128=.339/.298ms,N256=.451/.293ms. Five-expert data gives a weighted N64/N128 near-tie and N256 regression, NOT representative wholemodel tile selection. LogSHA `7fd70890dd210e01eba27fb6e1c0eca26de5e888d586319a382fe6747612b2a0`. Same widthprobe under ComputeSanitizer exits0 with0errors/0bytesleaked; logSHA `7df0b02f17b4b173c5ecd961788bb8fc75debbfab086b85210b522fa4d3c1668`; sanitizedtimings excluded.

Fullmodel uses SAME V25binary09526881.../source3e7f89b5.../checkpoint asV26. Exact candidate argv/env are in `/var/tmp/atlas-deepseek-vision-v27-n64-provenance.json`, SHA256 `1faba286e585721d2523e8bdf2abee6f302b34f0cded6c517090852de14e6dc4`. Add `ATLAS_EXL3_PREFILL_{DIRECT,PERSISTENT,FIXED_K2,FIXED_SHAPE,K64}=1`; N128/N256/M128/FUSED_POST/DUAL_PRE/FUSED_UNPERMUTE/FUSED_BLEND/W2A8=0 and ATLAS_PREFILL_MAX_REQUIRE_ARMS=0. KeepV26 eager/C1/native-sharedFP8/WOA0/CUBLAS0/prefixOFF, allcapture/HC-rounding flagsabsent. No watchdog override. The recordedPID is exited; freshbinding is mandatory before reuse.

| Working-model metric | V26 P1 | V27 P2 N64/K64 |
|---|---:|---:|
|256-token effective prefill, five-run median|30.9201 tok/s|346.1797 tok/s|
|256-token server TTFT median|8279.417ms|739.500ms|
|2048-token effective prefill, five-run median|161.9648 tok/s|706.4023 tok/s|
|2048-token server TTFT median|12644.724ms|2899.198ms|
|142-token exact passage decode, five-run median|18.0541 tok/s|18.1565 tok/s|
|165-token passage server TTFT median|8281.711ms|684.401ms|

All six text/color/image-order/repeatability outputs and all census/passage request/output hashes, token counts, cache0 and finishes match their working baselines. Root independently checked25 successful rawrequest/response/output hashes and finite timings. This is a first same-binary A/B candidate, not yet reverse-order/default promotion. Prefill is prompt/serverTTFT, not isolatedGPU; decode is API(output_tokens-1)/decode_time and essentially unchanged. No coding/DFLASH speed credit.

Smoke summary `/var/tmp/atlas-deepseek-vision-v27-n64-smoke/summary.json`, SHA900f99e0...; census `/var/tmp/atlas-prefill-census-deepseek-vision-v27-n64/summary.json`, SHA5d762adb...; passage `/var/tmp/atlas-deepseek-passage-decode-v27-n64/summary.json`, SHAd2926d92.... Complete hashes and crosschecks are in `/var/tmp/atlas-deepseek-vision-v27-root-review.json`. Basic smoke textTTFT534.80ms initially/247.59ms afterimages, single-image678.46–682.41ms, two-image1029.91ms; these are individual smoke samples, not image-performance medians. V7 encoder/JPEG/broadvision gates remain open.

Startup09:46:08→09:54:26 was slower thanV26; keep that separate from requestTTFT. Server log `/var/tmp/atlas-deepseek-vision-v27-n64-server.log`, SHA950bcd0a...; noCUDA/OOM/nonfinite/watchdog fault. Rootdrained boundPID1677695/start4571301, nativeexit130; postflight10:00:41 gives108GiB/noGPU/listener.

Next ordered work: fresh reverse P1 control with exact V26recipe and256/2048census/outputhashes; then fullmodelN128/K64 comparison, since the componentweightednear-tie cannot select the model winner. Optional BF16 post/dual/unpermute fusions require separate fullmodel gates; keep FUSED_BLEND off until its fresh actual fused-helper gate passes. W2A8 changes activation precision and must not inherit BF16 quality qualification.2000+prefill, VisionDFlash, broadvision and counting-watchdog falsepositive remain open.

## V26/V26B: measured working-model baseline

Same V25 binary09526881.../source3e7f89b5.../actualVision checkpoint and exact eager C1/native shared FP8/WOA0/CUBLAS0/prefixOFF recipe. Capture and HC-rounding flags absent; default watchdogs unchanged. Fresh provenance `/var/tmp/atlas-prefill-census-deepseek-vision-v26-provenance.json`, SHA256 `a5450a0ddaf23093dfc37c86a323079d327f3d707c0b4a367f450e77fd3ff28a`; binary/config/tokenizer/template/CWD/PID/start/argv/safeenv/listener pinned. Source was not edited. Fullmodel load09:15:43→ready09:21:35 is startup, not request TTFT.

| Workload | Warm / measured | Result | Server TTFT median |
|---|---:|---:|---:|
| Exact256-token fact prompt,32 output |1 /5|30.9201 effective prefill tok/s (30.7763–31.1108)|8279.417ms|
| Exact2048-token fact prompt,32 output |1 /5|161.9648 effective prefill tok/s (161.7577–162.1601)|12644.724ms|
| Exact passage copy,165 prompt /142 output |1 /5|18.0541 decode tok/s (17.8409–18.2498)|8281.711ms|

Prefill metric is prompt_tokens/serverTTFTseconds, not isolated on-GPU prefill. Server TTFT begins scheduler prefill handling, not HTTP arrival; clientwall retained separately. Decode is API `usage.response_token/s`, numerator output_tokens-1; do not substitute the server Done log's output_tokens numerator.8192, coding throughput/evaluation, DFlash/MTP and image-performance census are not measured here.

Existing frozen census `/var/tmp/atlas-prefill-census-v4.lnBBu3/census` SHA9646ca75... passed13/13 offline tests and ran unchanged with `--ids-mode prompt-array --prompt-format deepseek-chat --bins 256,2048 --timeout-seconds 600`. Summary `/var/tmp/atlas-prefill-census-deepseek-vision-v26/summary.json`, SHA256 `76cc0dab8e0375f8158722fcdefc5d239f000f89538838fe95ac9d2248aee318`. Canary exactlyATLAS_CANARY_OK; all12 fact outputs coherent, intentionally truncated at32tokens, byte-identical within bins. This is not full long-form factual qualification.

Longer decode companion reuses census's immutable provenance/transport modules, with complete exact-output/cache/finite/count guards. First count1..80 warmup FAILclosed: correct1..17,49tokens then default digit-normalized loop watchdog cut it off with finishlength (max256). Raw response `/var/tmp/atlas-deepseek-count-decode-v26/count-warm.response.json`, SHA256 `7ff5a4db4416e29bde8d380d5984ad0dd770ec638a091783b91491a3d658311b`; no performance credit. Native `scheduler/decode_logits_content.rs` calls the numeric-normalized detector, which treats changing numbers as the same token. This falsepositive remains unresolved; no watchdog override was enabled.

Measurements then stopped while root prepared a new nonrepetitive passage-copy companion in `/var/tmp/atlas-deepseek-passage-v26b.vHjSKn`:364LoC sourceSHA2fca4cfd...; binarySHA7d812488...; expectedTDDRED then5/5CPU/-Dwarnings/fmt passed. The same server remained idle during bounded CPU preparation; all CPU work ended before the new warmup. Only workload/framing/labels and one framing test differ from the frozen count kit; safeguards unchanged. New summary `/var/tmp/atlas-deepseek-passage-decode-v26b/summary.json`, SHA256 `e480157322881385bc57a4c9e97921392cf0480250b3d2b9ae3f659751dcf666`. All6 responses exact complete passage/natural stop/cache0; outputSHA `16f6b562a028482002975268fbe441ab878074a166a035944ce95d7cae3bf59e`.

Root independently verified all19 successful rawresponse/request/output hashes, token/cache/finish/content and positive finite timing fields. TTFT raw-to-record JSON roundtripping differs by one ULP in some records; crosscheck allows only4.5e-16 relative serialization roundoff, not a model-parity tolerance change. Separate rootreview `/var/tmp/atlas-deepseek-vision-v26-root-review.json` preserves immutable driver semantic-status fields. Server logSHA `1620dfa7069852c3acf9e5690fa323b9ed51b02bdf77193a5e129d52361cd5b0`. No CUDA/OOM/nonfinite fault or competing workload; exact ownedserver drained09:31:24, exit130, port/GPUempty.

### Next performance gate, not an implemented win

Read-only P1 source audit: `forward_prefill_exl3.rs` dequantizes all experts in any active scratch chunk; absent `ATLAS_EXL3_PREFILL_CHUNK` uses8. The pinned V20 L0 twelve-token routing capture has65 unique experts but chunk8 touches224; chunk1 would touch65, with more launches. This is a concrete short-TTFT candidate, not an established whole-model bottleneck or speedup. Fresh resource claim, same-output/vision gates and repeated timing are required. Existing direct/P2/W2A8 paths are another candidate but not yet qualified for the actual Vision checkpoint. No model default/quantization or performance tolerance changed.

## V25A full-model semantic result: basic text and vision work

Six cases, in order, all returned exact expected content and stopped normally: `OK`, `RED`, `BLUE`, repeated `RED`, two-image ordered `RED, BLUE`, then `OK`. Repeated red and before/after text output hashes are identical; allcachedtokens0. Same actualVisionEXL3checkpoint/C1eager/nativeFP8/WOA0/CUBLAS0 recipe as V21-A, with HC-rounding and allcaptureflagsabsent. Driver completed0; no CUDA/OOM/nonfinite fault found. Root checked PID/exe/start4212149/port before graceful SIGINT; nativeexit130 and emptyGPU/listener afterrun.

Summary `/var/tmp/atlas-deepseek-vision-v25-a-smoke/summary.json`, SHA256 `8d572930f7ae598bf9af4419d0adf0dbd5c656368c78307d03796beffca5a78e`; provenanceSHA `ac5a8bc472052b21f7a445a8cc10ca25e5295fa17c73e1a4c075ded4bd3c38bb`. Separate rootreview `/var/tmp/atlas-deepseek-vision-v25-a-review.json` records the narrow pass without mutating frozen driver output. The syntheticRGB336 PNG cases prove basic color perception, image ordering and request-state isolation, NOT broadmultimodal/OCR/JPEG/fullencodernumerical qualification. V7 encoder parity failure remains explicit; tolerances unchanged. NoVisionMTP/DFlash qualification.

Observed smoke timings only:12token textTTFT5037.0ms then4797.5ms;137token one-imageTTFT7528.1-7593.3ms;271token two-imageTTFT8743.3ms. Server-reported decode17.69-18.18tok/s has only2-5 generatedtokens and is not representative. Keepimageencoding/clientwall/serverTTFT/on-GPUprefill separate. Longer textdecode and256/2048prompttoken census require fresh resourceclaim and preserved exactprovenance;2000+prefill remains unachieved.

## V25 GPU regression result and full-model retest

Root read all5 final TEMP probe sources plus pinned legacy driver/ABI/metrics and independently reran7/7 CPU tests. Admission `/var/tmp/atlas-v25-blend-probe.jUafF2/admission-v1/admission.json`, SHA256 `450fd36fc6a504686d5285c75133b336930ec7e9668263a071ca5341360ff674`, reclosed with every payload before GPU. Exactly10 cases: capturedM12/M1row0 and syntheticM1/M12 with NULLgate/nonNULL dot0/+1/-1, all256thread coverage.1,646,592ownedGPUbytes (<4MiB); no fullmodel resident.

Old PTX replay exits2 as expected:10/10cases fail,255badthreads percase; capturedM12 rawoutput is byte-exact to the fullmodelV23 output. Result `/var/tmp/atlas-v25-blend-probe.jUafF2/gpu-old-v1/result.json`, SHA256 `36c66e9486d22bdd0b065b4f38d58def5bb8e422ec2bc16d0aedb8fc8c430b76`. Corrected PTX exits0:10/10exact,0BF16differences, allinput/shared/gatebuffers unchanged, successful cleanup. Result `gpu-candidate-v1/result.json`, SHA256 `da090f8b3c3342a7a5fd3e95aa8e15666506af7d15ce15851e7813de81f5e305`.

Corrected ComputeSanitizer memcheck replay exits0 with0errors/0bytes leaked. All36 rawoutput/unchangedoperand files byte-identical to unsanitizedcandidate. Result `gpu-candidate-v2-memcheck/result.json`, SHA256 `44eed5743d4251f16bee62d4173a11ec326e6552b02e04721e50657a3933082f`; log `/var/tmp/atlas-v25-blend-candidate-v2-memcheck.log`, SHA256 `226ed4692a069eb335e3d1fd73c0e9e58e5bfd29fc183f0339358200b1e2eb52`.109GiB/noGPU/listener after boundedreplays. This proves the concrete standalone blend correction, not fusedtail/fullmodel vision correctness or speed.

Fresh fullmodelV25A claim08:45:09 uses binary09526881... and exactV21-A settings, HC-rounding and ALLcaptureflagsabsent. Server log `/var/tmp/atlas-deepseek-vision-v25-a-server.log`; provenance `/var/tmp/atlas-deepseek-vision-v25-a-provenance.json`, PID1578948/start4212149 (verifylive before reuse). Only six-case text/red/blue/repeatred/interleaved/text smoke is claimed, existing frozen driverf4a9f588...; source and childCPU remain paused whilemodelresident. Semantic results pending; wronganswersmustnotbe promoted. Drain exact process before another physical window.

## V23 result and V25 first-divergence correction

Actual V23 one-token request completed08:23:57: HTTP200,12 input tokens/1 output/0 cached, firsttoken201 (empty rendered content, still wrong). Both captures are COMPLETE. Native PID1524373/start4037310 was rebound after request and gracefully drained, exit130. MoE manifest SHA256 `9fc430a5411af9660970c593843397a36ed6742c15cc50345ed2698b2d4f1bdc`; CPU check `/var/tmp/atlas-deepseek-vision-v23-moe-check/report.json`, SHA256 `fa98e94650d9cb35d0baf07cd1bdb1ec8a178c3d71096e561ca13484ed6f9159`, exit2 denotes valid capture with failed relationships.

All6 native FP8 weight/F32 scale payloads match frozen checkpoint identities. Outer FFN and inner shared input are byte-exact; shared-down and shared-after-routed are byte-exact. New HC capture agrees exactly with new MoE input/output. Blend fails48587/49152 BF16 elements, aggregate relativeL2 .1581906223. Raw operands prove ONLY columns divisible by256 change from routed-only, and those changes are the correct shared addition. Input SHA `096fc88eff84aba38341bd01c702c5c6ea5dadc035512d5c5cdbe176e172f17f` differs fromV14, so previous activation references are not interchangeable.

Root and independent agent traced the actual selected `moe` module to immutable V23 `t0__moe_permute.ptx`, SHA256 `759aed5ba5906f65c01ea89298a6454614b893bc163db0188c2fdeec1f0299ea`. In `moe_batched_blend`, nonzero threads load sharedgate atline942, BEFORE tid0 publishes gate=1 at969 and secondbarrier972. No reload follows; stale zero is multiplied into shared output at993. Source helper expresses the read after synchronization but uses nonvolatile restricted shared scratch. Module selection and4pointer+2u32 ABI are correct. The exact observed column pattern supports this compiled broadcast diagnosis.

V25 claim08:28:12: minimal `moe_batched_blend.cuh` scratch parameter volatile/removerestrict plus focused TDD, preserving all arithmetic and ABI. Header SHA256 `a1f509d79c9296aba50160b792f89037739b98b7425938b9d970c98859d35cb7`; reverse-transform exactly reproduces oldheader9103c340...8175e9. Tests:356 model library+31 integration, including9 fused-blend and22 existing capture/native/HC/KV, allgreen after expected TDDRED. Source fingerprint `3e7f89b5b5d0acb4ae4ef8a79fbad3756a0cc805ab3265823fe6133ea493fbfc`.

Fresh native build `/var/tmp/atlas-deepseek-vision-v25` passed207modules/35overrides in1m03s. Binary SHA256 `095268810321ab8760c570adb7c9f9e8e5544958f771dcee246cf183404d83d8`. Only `moe_permute`, `exl3_gemv_k2`, and `exl3_w2a8_h128_emit` PTX changed; other204 are byte-identical toV23. Root verified compiled standalone and fused helper publication store, secondbarrier, THEN shared volatile load. Candidate `t0__moe_permute.ptx` SHA256 `7e4c686111f772cd8107ebd1814dbf72af2e8f444fca60f664db6947db66e3fc`. Source/binary gates are not GPU/model qualification. TEMP probe `/var/tmp/atlas-v25-blend-probe.jUafF2` is preparation only; captured/synthetic NULL/nonNULL GPU replay and full-model semantic retest remain. Do not call DeepSeek fixed until semantic text/image gates pass.

## V23 capture implementation and provenance

Default-absent `ATLAS_VISION_MOE_L0_DUMP` observes the actual L0 native shared input, three FP8 weights/F32 scales, gate/up/clamped activation/down, preserved shared output after the entire routed path, routed-only output, and blended output. Exactly15 stages plus pinned12 IDs total25,909,296B; CPU copies stream in512KiB chunks, no GPU writes or allocations, COMPLETE manifest last. Unsupported modes fail closed. Existing HC capture schema/8MiB limits are preserved via extracted filesystem helpers.

CPU:356 model library +22 focused integration; root independently verified46 server library +820 binary tests (one existing ignore), and16/16 independent capture-consumer tests. Native build passed207 unique modules/35 overrides in1m02s; all207PTX byte-identical toV18. Source fingerprint `81be39fb9fb21a6ee77e32ed774bacfac7769df4c54b06257a03b40331ae5f3c`; binary `/var/tmp/atlas-deepseek-vision-v23/release/spark`, SHA256 `f3d0e8f23de0aebfbd5db4ccf1bf5d035b431b9bea9a7bdbd9dae96c3af3a6a1`.

Root claimed GPU/server8977 at08:16:30 for ONE exact12-token maxout1 request, V21-A recipe (`WOA_INPLACE=0`, `CUBLAS_TUNED=0`, HC-rounding flag absent), new MoE capture `/var/tmp/atlas-deepseek-vision-v23-moe-capture` and old HC capture `/var/tmp/atlas-deepseek-vision-v23-hc-capture`. Provenance `/var/tmp/atlas-deepseek-vision-v23-provenance.json`, PID1524373/start4037310; verify live before reuse. Do not compare activation tensors toV14 without establishing identical inputs. Drain model before independent GPU reference. Synchronized capture earns no performance credit.

V22 sanitizer replay `/var/tmp/atlas-native-shared-probe.XIesrt/gpu-v2-memcheck/result.json`, SHA256 `2ed623d4875ae43a572d9384d0d54ada0c2772b71e58f06cd86ba3245ab76e56`, exited0. Installed Compute Sanitizer2025.3.1 reports0errors and0bytes/0allocations leaked; all9 raw stage outputs byte-identical to unsanitizedV22. Log `/var/tmp/atlas-native-shared-probe-v22-gpu-v2-memcheck.log`, SHA256 `141d94d923b23f846ae489095e805dd11cb3d9d9642c010cfbdcdb89a82b65f9`. This covers only the bounded native shared probe, not full-model memory safety.

## Exact target and runtime

- Checkout: `/home/flocka/atlas/apathy-deepseek` (dirty shared worktree; preserve unrelated changes).
- Actual Vision quant: `/home/flocka/models/DeepSeek-V4-Flash-Vision-EXL3-K2-c171bea5`.
- HF: `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1`, revision `c171bea574201ff25530256fbd63626c7fd20f3c`.
- Config SHA256: `28a07138554196d7de70cfb193eb63bf51c39bb42ae4cd4303ba16610b5b1bf5`.
- Ten shards, 84,913,196,616 tensor bytes; native BF16 visual tower/aligner/special embeddings: 267 tensors, 932,786,176 bytes. These are actual continued-trained Vision weights, not a text checkpoint plus a sidecar.
- Native V6 server: `/var/tmp/atlas-deepseek-vision-v6/release/spark`, SHA256 `7a1eea4b858cc5564cacd72a62f5a62345d706586865717fab2570c705267a76`.
- Source-input fingerprint: `f2fe6dfe263356fe61565b3695b6057e02731972d07b4c9f084a826ecdaa02e3` (Cargo/config/source/kernel/template inputs; excludes this note).
- Native build: 206 unique modules, 34 overrides; all CPU library tests 328/328, server 820/820 plus one existing ignore, KV guard 9/9, probe 4/4.

## Implemented contracts and restrictions

Typed Vision config and checked CSA/HCA layer labels; exact image sentinel/N-layout grammar; RGB8 PNG/JPEG data-URI preprocessing; dedicated 32-layer RMSNorm/SwiGLU/2D-RoPE ViT and 3x3 aligner; safe image embedding splice; per-token `bias_vl` expert routing; initial-chunk bidirectional image visibility.

Only single-GPU, C1, target-only serving with prefix caching disabled is admitted. Every image must end in the initial prefill chunk. FP8 KV only; generic BF16 and existing NVFP4 V4 decoders lack required hybrid-attention semantics. Vision graph replay is disabled because compressed-history updates require eager execution. Keep `ATLAS_FP8_KV_EMA_RECAL` absent. BF16 lm-head is used. JPEG decoder numerical parity is not qualified; first image tests use RGB8 PNG.

## Encoder evidence: unresolved

`/var/tmp/atlas-deepseek-vision-parity-v7-20260905T0440Z` contains exact input/output hashes, repeated native runs, bounded untimed stage captures, and pinned official comparisons.

The FP32-probability/PV candidate is deterministic but **fails the predeclared full-encoder gate** at 4x5 and 54x54 grids. Original thresholds remain cosine >= .999, worst-row cosine >= .995, relative L2 <= .05. Do not advertise this as passed.

With the pinned official math SDPA, worst-row cosine is .986739 at 4x5 and .974043 at 54x54. Independent shared-input operator tests are much closer. A shape-dependent Torch reduced-precision GEMM explains two larger local differences: disabling that reduction makes 4x5 block0 FC1 and aligner W1 byte-identical to native, but does not close full-encoder parity. Full-precision shared-input operator/block worst-row cosine is at least .9999839. Distributed rounding amplification remains unresolved.

Official reference revision: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@6821d6ad3681a4b137b066b76094fa82ebd0a380`. Temporary reviewed oracles: `/var/tmp/atlas-deepseek-vision-oracle.ONOuQa/compare_reference.py` and `compare_stages.py`.

## Bounded full-model diagnostic V8

Startup reached all 43 layers but failed the memory-budget check before HTTP: 100.6 GiB pre-KV + 0.5 GiB reserve exceeds the .84 utilization cap of 100.5 GiB. This was a controlled refusal, not CUDA OOM. PID 1098758 exited; no image request ran. Provenance: `/var/tmp/atlas-deepseek-vision-v8-provenance.json`. Log: `/var/tmp/atlas-deepseek-vision-v8-server-retry1.log`. Next bounded launch will use .87 with the same fixed KV cap after V9 CPU/native gates.

Recipe: FP8 KV, BF16 lm-head, max sequence 12288, max prefill 2048, KV cap 12304 (includes dummy-block margin), utilization .84, 2 GiB OOM guard, C1, prefix cache off, thinking off, explicit eager decode. `--no-tui` is unsupported by this DeepSeek CLI and was removed after a pre-GPU argument rejection.

Next gate: six-case `/var/tmp/atlas-deepseek-vision-smoke.QpwqI1/vision-smoke` (text OK; red; blue; repeated red; interleaved red+blue; text OK). Preserve all responses and review exact answers and state isolation. These simple synthetic tests are not broad multimodal qualification. Encoder qualification failure must remain explicit even if image answers pass.

## Native shared FP8 candidate V9

Default-off `ATLAS_V4_SHARED_NATIVE_FP8=1` preserves the actual shared-expert native E4M3 weights and widens E8M0 block scales for existing W8A16 GEMV/GEMM. Both prefill and single-token decode use unclamped shared SiLU; routed EXL3 kernels are untouched. Valid NVFP4 fallback storage is retained, with only 258 KiB additional scales. Requires actual Vision, EXL3, single GPU, eager target-only operation; wide verify is rejected.

Focused 5 unit + 2 integration + 9 KV guards passed. Root fresh full CPU gates: model 338/338, binary tests 46/46, server 820/820 plus one existing ignore. Frozen source-input fingerprint `2f64b5f6f008d80764429187f65588669b50a25446e6552ca1333e366db8841d`. Native release build in `/var/tmp/atlas-deepseek-vision-v9` passed in 1m00s, 206/206 unique CUDA modules and 34 overrides; source fingerprint remained identical. Server SHA256 `eb062a137987c648274e734865bdc49f7e30b3f7cadc549c7f0733c08a479483`. This is not yet GPU-qualified.

V10 full-model startup passed: all 43 native shared FP8 layers armed, pre-KV 100.5 GiB, 0.6 GiB FP8 KV allocated, HTTP on loopback 8977. Same bounded V8 recipe except utilization .87 and `ATLAS_V4_SHARED_NATIVE_FP8=1`. Log `/var/tmp/atlas-deepseek-vision-v10-server.log`, provenance `/var/tmp/atlas-deepseek-vision-v10-provenance.json` (PID 1159433 is now exited, do not reuse).

The six-case smoke completed but **0/6 semantic expectations passed**. Text `OK` instead produced an unrelated DSML/anchor explanation; red produced `<dsfsad`; blue produced repeated `<ds`; interleaved red/blue claimed text was too small to read. Repeated red and before/after text outputs were byte-identical, but this is not quality. All responses were complete/finite/uncached, with no CUDA fault. Evidence: `/var/tmp/atlas-deepseek-vision-v10-smoke-20260905T0521Z/summary.json`. Reported decode about 17.6-18.5 tok/s and image TTFT 7.77-8.63 seconds are **failed-output diagnostics, not valid performance results**. No long-prefill benchmark was run against these failed outputs.

Owned driver exited normally; server was drained by SIGINT and exited 130. GPU inventory empty and 108 GiB preflight verified afterward. Root moved GPU ownership to Flash-Next census C3. DeepSeek source remains frozen; read-only decoder/template/precision diagnosis is active. Because text-only also fails, encoder mismatch alone cannot explain the full failure; no sole cause is established.

## Confirmed V9 contract mistake and V11 correction

Recovered exact official decoder `/var/tmp/atlas-deepseek-official-6821.vG09SK/model.py`, SHA256 `ae8de79223d53c16edc7fa67af124ddd0212f27bddd3207bdd4cb05b48a0a1a7`, proves at lines 655-658 and 682 that the actual Vision **shared expert also uses `swiglu_limit=10`**: upper-clamp gate at 10 and clamp up to [-10,10] before SiLU/multiply. The earlier V9 unclamped-shared premise was wrong for this reference. V11 changes only the native shared activation selector to existing `moe_silu_mul`, with pinned-reference regression/boundary tests and corrected diagnostic text. No CUDA/routed/old-text changes. This is a confirmed contract correction, not yet proven sufficient to fix the failed canaries. V10 artifacts remain preserved; new CPU/native build and V12 identical-case smoke are required.

V11 gates completed: two expected TDD failures, then model 338/338 + integration 4/4 + server 820/820 (one existing ignore), formatting/diff clean. Fresh native build passed 206/206 modules, 34 overrides, 1m03s. Source fingerprint `9f5a7d203aa50f4bd41d27d58a483a9353b5f1a505b608b4c612b5bd88417155`; server `/var/tmp/atlas-deepseek-vision-v11/release/spark`, SHA256 `951421f8a8b78a5642d3eb73a0d64de537ff155f19507ed98f4fffd8691969e8`.

V12 started 05:35:24, all 43 clamp10 ARMs and HTTP passed. Same V10 runtime arguments/environment except binary. Log `/var/tmp/atlas-deepseek-vision-v12-server.log`, provenance `/var/tmp/atlas-deepseek-vision-v12-provenance.json` (PID1204041 now exited, do not reuse). Fixed limit10 matches this pinned checkpoint; arbitrary `swiglu_limit` configs are not qualified.

V12 six-case smoke again **0/6 semantic matches**: text now generates an unrelated math solution, red `>`, blue `<code>object</code>`, interleaved unrelated text. Outputs changed from V10, but repeated red/text are still byte-identical. The clamp correction engaged and is not sufficient. Evidence `/var/tmp/atlas-deepseek-vision-v12-smoke-20260905T0541Z/summary.json`; both text first-prefill IDs are33001. Driver completed0; owned server drained with SIGINT/exit130; GPU and listener were verified empty. No valid DeepSeek performance benchmark.

## V13 first-layer localization in progress

Root reserved a default-off, actual-Vision, text-only, C1, first-chunk, layer0, exact12-row diagnostic. `vision_moe` owns a new bounded typed capture module and minimal HC-prefill hooks/tests; no numerical/CUDA changes. Capture full BF16/F32 rows and stable IDs, <=8MiB, fresh fail-closed output directory, no overwrites, final complete manifest only after all payloads succeed. Generic `ATLAS_OP_DUMP` is bypassed by this HC path and must not be mistaken for a valid HC capture.

`deepseek_vision_contract` independently prepares a temporary same-input HC/RMS reference using only about3MiB selected HC/RMS weights, never a full model. Native capture and small reference must run sequentially with a drained native server between. A confirmed precision-contract difference is official HCpost returning BF16 while Atlas keeps FP32 residual streams; it is a diagnostic target, not an established cause. Compare raw native FP32 and BF16-rounded native against reference before proposing a correction.

Official sources are pinned in `/var/tmp/atlas-deepseek-official-6821.vG09SK/`: `model.py` SHA256ae8de792..., `kernel.py` SHA256b9e91e297c26e586a3fcdc8af304436d7d7c13159a876dc7e6fa2e76a8055a78, `config.json` SHA256be1af50a0a8dc4e2e096362f9cc79d821846075a45863ded0b7d9bf4512075f6. Root read the HC/Sinkhorn/RMS formulas and actual entrypoint mapping: inference config explicitly `norm_eps=1e-20`, passed through `ModelArgs(**json.load(config))`; it agrees with checkpoint `rms_norm_eps=1e-20`. Do not change epsilon to the unused dataclass default1e-6.

Root-only one-token request kit `/var/tmp/atlas-deepseek-v13-request.HXk3pa/` is prepared but not run. Exact same12-token OK prompt as V12, generation budget1; require first-prefill ID33001 and captured IDs equal separately tokenized prompt. No throughput credit from synchronized captures. No GPU/server/native build authorized until both diagnostic CPU lanes freeze and root reviews.

## Fresh prefill census

### Latest DeepSeek diagnostic evidence V14-V16

V13 native diagnostic build passed206 modules/34 overrides; source fingerprint `a2ec9f8f845c52309481646437e323ee8c64b3863d31756b390cb5f75f1945d6`, server SHA256 `d689c9964dbeec7b35daec29ec8065e056f59c33826bf5bb9e827dc61f4b6f45`. All kernel build outputs are byte-identical to V11. Capture CPU tests345 model,2 capture integration,4 clamp integration; server820 plus one old ignore.

V14 one N12/maxout1 request returned HTTP200/uncached but first11655 (`######`) versus V12 first33001. Capture transparency is **not proven**. All14 typed tensors plus IDs completed3049392B in `/var/tmp/atlas-deepseek-vision-v14-l0-capture`, manifest SHA256 `8539c7daca1b44e8ea380d634142dacfb498b8820bf898779bca4696c8d15861`; separately tokenized IDs match. The12 captured embedding rows are also byte-identical to actual checkpoint rows (SHA25650042963...). Native server1253819 drained/exit130; empty GPU verified. Responses/transport in `/var/tmp/atlas-deepseek-v13-request.HXk3pa/v14-*.json`.

V15 reviewed small HC/RMS reference ran alone on GPU and exited0, selected8 weights3162328B. `/var/tmp/atlas-deepseek-vision-v15-hc-reference/manifest.json` SHA256 `c0b3329c3bff8bfd06f1125277b7b60e7d61ecee1fd796547c46a0306b80a651`. Expansion/HCpre-attn/RMS-attn exact. Same-input FFN HCpre/RMS each differ at2 of49152 BF16 elements; comb relativeL2 about1.1e-6. HCpost FP32 arithmetic relativeL2 below6e-8. Official BF16 boundary is distinct but no catastrophic HC/RMS error is demonstrated. Attention and MoE remain unverified; no chained/full-model parity or performance credit.

Read-only tuning audit found grouped wo_a cuBLASLt12x1024x4096 (lda32768,ldc8192) selected heuristic2 in V12 but1 in V14; strided path autotunes even with ATLAS_CUBLAS_TUNED absent. Capture adds default/prefill stream barriers and host pauses, so effective numerical algorithm is not frozen across loads. No direct capture write/alias bug found; max_tokens does not enter first-token sampling. V16 used identical V13 binary with capture absent and the same C1/FP8/eager recipe. Four serial maxout1/32/1/32 requests all returned first token33001; both max32 outputs reproduced V12's unrelated math answer byte-for-byte. Heuristic2 was selected again. Output-limit changes therefore do not explain V14's different first token in this control; tuning correlation is not proof of a sole cause. PID1295824 is exited, session27416 terminal130, GPU/listener cleared. Provenance `/var/tmp/atlas-deepseek-vision-v16-provenance.json`; responses in `/var/tmp/atlas-deepseek-v13-request.HXk3pa/v16-*-response.json`.

V17/V19 bounded L0 attention references completed with native model drained. All use the exact V14 `norm_attn` input, 13 selected weights106964608B, and pinned official source; no full-model load. Official FP8 execution relativeL2 versus native attention output is .03406175. A declared BF16-projection counterfactual reduces this to .02279419; additionally matching native F32 Q normalization gives .02276334. Matching native unquantized local KV reduces it to .00510765 (worst-row cosine .99998429). Thus missing official K64 E8M0/FP8 QAT accounts for most remaining local attention difference, not proven whole-model failure. Original official result and all counterfactuals are preserved separately under `/var/tmp/atlas-deepseek-vision-v17-attention-*` and `v19-attention-*`. No tolerance widening or full-model correctness/performance credit.

V18 implements explicit-default-off `ATLAS_VISION_HC_BF16=1`: actual typed Vision only, H4096/HC4/C1/singleGPU/target-only eager. New isolated HCpost kernel retains the existing seven-argument ABI, FP32 arithmetic/storage, and changes only final store to BF16-RNE then F32. Original `hyper_connection.cu` unchanged. Source fingerprint `031a4ffddf42e04c5fb48744310c6656395fe3ffa15f0bc8a75ba79c913efafa`. Model351/351, focused integration20/20, server library46/46, server binary820/820 plus one existing ignore. Fresh native build `/var/tmp/atlas-deepseek-vision-v18` passed in1m03s:207 modules/35 overrides; all old kernel artifacts match V13 byte-for-byte. Server SHA256 `835a0cb194411b062fb41e3dc104357e1fa0bd053c5ca44ae2c313383beda281`. Standalone GPU probe passed64 exact checks: old kernel reproduces V14 HCpost captures, new kernel equals independent BF16 rounding across synthetic/captured, N1/N12, alias/disjoint, grid.y1/16 controls. Probe log `/var/tmp/atlas-deepseek-vision-v18-hc-probe.log`, SHA256 `788164bd9acd94481df1be14768bb064e8797721de99b15497d7257143d0dcca`. This is boundary-only parity, not a model fix.

V21 uses the same V18 binary for separate baseline/candidate full-model startups. Both set existing `ATLAS_V4_WOA_INPLACE=0` and `ATLAS_CUBLAS_TUNED=0`, use the same V16 C1/eager/native-sharedFP8/FP8KV/.87 fixed-cap recipe, and disable capture. The gathered projection path avoids timing-selected strided GEMM drift but is not claimed equivalent to V16, hence its own baseline. A has HCflag absent: all43 shared ARMs, HTTP ready07:11:06, six requests completed, **0/6 correct**. Text generated unrelated C includes; red `ull`, blue a DSML fragment, interleaved alphabet. Repeated red/text hashes match exactly. `/var/tmp/atlas-deepseek-vision-v21-a-smoke/summary.json` SHA256 `34ad1e1a68de2dec7b69a67953dbf4e26dbf1b5a4e7be1ddb6b6d6db20d8b3fa`; matching server log/provenance retain effective state. PID1384412/start3605990 was drained (session8626 exit130);109GiB and empty GPU/listener verified. Candidate B with only HCflag1 added remains pending. No model quality or performance credit.

V21-B completed with43 HC-rounding ARMs and43 native-shared ARMs. Same binary/config/argv as A; only `ATLAS_VISION_HC_BF16=1` added. All six requests completed, **0/6 correct**. Text now generated unrelated Python imports; red/two-image outputs match A, blue changed to another DSML fragment. Repeated text/red remain deterministic. B summary `/var/tmp/atlas-deepseek-vision-v21-b-smoke/summary.json` SHA256 `8b1e3419991181ff1f9bc9d28c49086dd628b3b87f8c030676358593e73c537f`; provenance SHA256 `331e249e12d683706957981965334ca7c4e08c6ad0721c8891bcb484dbfca6f1`. No CUDA/error marker. PID1403403/start3666730 drained by SIGINT/session20952 exit130;109GiB/noGPU/listener verified. The boundary correction is not sufficient; it remains default-off with no speed/quality credit.

Next diagnostic is the independently implemented selected-expert MoE reference in `/var/tmp/atlas-deepseek-moe-oracle.j1PZ1M`. Root read final source/tests and verified their hashes against frozen admission. CPU tests21/21, admission/readback exit0:65 selected experts from72 assignments,787 selected tensor receipts. Admission `admission-v1/admission.json` SHA256 `071b711e7b7bd5be13caece38b7c6bb643234b0bcac1b7ed1b060c8da3664102`. After a fresh root GPU claim, run `oracle.py run --admission admission-v1/admission.json --admission-sha256 071b711e7b7bd5be13caece38b7c6bb643234b0bcac1b7ed1b060c8da3664102 --arm decoded-math --device cuda:0 --allow-gpu --out /var/tmp/atlas-deepseek-vision-v20-moe-decoded`, then separately `--arm v14-p1 --out /var/tmp/atlas-deepseek-vision-v20-moe-p1`. Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/python3 -B`; only one16MiB decoded projection resident,512MiB PyTorch tensor cap, no full-model/weight conversion files. Both arms are declared EXL3 diagnostics, not original-FP4 or full-model parity. No MoE GPU result exists yet.

### Completed V20 MoE comparison and V22 next gate

The preceding V20 pending-run paragraph is superseded: both declared arms ran sequentially on the idle GB10 and exited0 by07:26:31. `v20-moe-decoded/comparison.json` SHA256 `554cf4c2eaba7ecef42d039570a4617c3defe602bcccd46b762d0f8d1a688494`; `v20-moe-p1/comparison.json` SHA256 `cd9484c5a203f00175023c87082908399a22c383ffbb374eca11ad3ee6daabc5`, both under `/var/tmp/atlas-deepseek-vision-`. Peak Torch allocated84.68/67.87MB, TF32 and reduced-precision reductions off. Each comparison preserves669 stage receipts. These are independently implemented EXL3 diagnostics, not original-FP4 or unmodified-donor parity.

Aggregate relative L2 .160870/.161539 is dominated by the huge BOS output. P1 row0 relative L2 .022336, but ordinary rows1..11 range .885955 to5.245107, with cosine .329 to.885. The mismatch is substantial, not a uniformly small rounding error. CPU decomposition shows native output close to routed-only (cosine .97277 to.999997); native-minus-route norm .408 to1.152, while predicted shared norm6.543 to20.411. Native-minus-route direction cosine versus predicted shared is only .041 to.079: a scalar gain correction would not resolve this.

Read-only audits found no dropped shared call, scratch alias, suppressed shared blend, or freed original FP8 weight pointer. `quantized_from_fp8` frees only its newly allocated BF16 intermediate, retaining original FP8 storage. Actual shared scales contain only E8M0 codes115/116, excluding the discovered zero-code conversion edge as an explanation. No production numerical patch is justified by this evidence alone.

V22 `/var/tmp/atlas-native-shared-probe.XIesrt` will run immutable V18 W8A16 GEMM/SiLU PTX directly on the actual three layer0 shared FP8 weights and V14 `norm_ffn`, comparing gate/up/down-input/down output to V20 saved stages. This separates kernel math from full-model loader/scratch/dispatch. Initial8 CPU ABI/admission tests passed; final chain/admission/root review and fresh GPU claim remain required. Device-data cap32MiB, no full model, no throughput credit. ModelForge audit is read-only and does not authorize checkpoint conversion or model surgery.

V22 final CPU tests10/10 and frozen admission-v2 SHA256 `969f2e7fa30c7a5c7acc434604993979f7ac09396040feb64ab1e12900e5a064` passed root review. Native GPU run completed07:45 with successful cleanup and25,630,720B owned device allocations;109GiB/noGPU postflight. `/var/tmp/atlas-native-shared-probe.XIesrt/gpu-v1/result.json` SHA256 `90c05bb8860d1832ed83b37b4caf327356920213b6690623ec162d8dd54c6012`. N12 gate/up/activation/down differ from V20P1 by1/3/4/4 BF16 elements, relative L2 2.39e-10/2.40e-7/6.32e-8/5.11e-6. Down-only control with saved reference activation has the same tiny error. N1 GEMV row0 gate/up/activation/down are all byte-exact. This does not reproduce the large full-model shared-output discrepancy; inspect actual full-model weight/input/output and preservation across routed execution next. No model-quality or throughput qualification.

### Existing-tool reuse audit

Root inspected and ran existing ModelForge header-only `map ... --format json` with bytecode disabled and stdout only. Raw report preserved at `/var/tmp/atlas-modelforge-vision-inventory.9h9SlP/map.json`. It reports43 layers but `is_moe=false`, `num_experts=0`, `active_experts=0`, and `quantization=null`, while the actual config explicitly says256 routed experts, top6, `quant_method=exl3`. Per-layer reporting itself finds256 experts. The generic mapper also folds visual layer names into decoder accounting and lacks F8_E8M0 byte-size support. This output is an approximate inventory, not admission or a reason to modify weights. CLI/source hashes: cli.py `c7ee0ad179e4cfc2bb2bbe484e093c8468963763e3d1e4f75ad59fe536240c46`, commands/map.py `57b166dcc27eae0b48a404613e1272b19f26b04d76629a5db941fcbc67c5eeef`, model_discovery.py `2c92a8f4360c296145c4139cb0b89f85b9cbf87321e5141ae7262250e1848110`.

ModelForge `diagnose static` has reusable strict admission infrastructure, but its current catalog rejects nested `.cache`/`encoding` directories and its generic vocabulary/quantization adapter does not establish DeepSeek embed/head or EXL3 assignment semantics. It was not run against84.9GB weights. Existing diff/batched-inspection commands are not a bounded native EXL3 numerical reference. Keep pinned selected-weight oracles for immediate diagnosis; no ModelForge adapter/quantization/surgery lane is authorized by this audit.

Metric below is exact input token count divided by server time to first token, **not isolated on-GPU prefill**. C1, uncached, temperature zero, 32 requested output tokens. Do not combine target-only prefill with DFlash decode into a single configuration claim.

| Target/mode | 256 tokens | 2048 tokens | 8192 tokens |
| --- | ---: | ---: | ---: |
| GLM 5.3 Flash R55, target-only layer-major | 316.49 tok/s | 459.99 tok/s | 105.88 tok/s, one sample |
| Actual DeepSeek Flash Vision | Withheld: text/image correctness failed | Withheld: text/image correctness failed | Not measured |
| Qwen 3.8 27B target-only, ChatML | 307.39 tok/s | 320.58 tok/s | Not measured |
| Qwen 3.8 Flash-Next NVFP4 control, ChatML/QSA | 43.21 tok/s | 41.80-41.83 tok/s, two samples | Not measured / not qualified |

GLM short bins each have one warm plus five measured samples. At 8K, warm TTFT was 77.024 s and the first measured TTFT 77.369 s; remaining repeats were stopped to prioritize Vision. This is a partial census, not a five-sample 8K median. Default-thinking chat canary exhausted its budget instead of returning the literal, and raw summaries are not a factual/coding quality qualification.

Evidence: `/var/tmp/atlas-prefill-census-glm-r55-20260905T0430Z/partial-review.json`. Owned GLM server and driver exited. Vision-kit compilation overlapped only excluded 8K warmup; completed short bins and first measured 8K ran outside that compilation.

Qwen V3 ChatML canary returned exact `ATLAS_CANARY_OK`; 256/2048 each completed one warm and five measured, cached tokens zero, measured outputs byte-identical within each bin. TTFT medians 832.815/6388.448 ms. Short-run server-reported decode medians were 14.01/13.76 tok/s, not a DFlash result or a standalone decode benchmark. Summaries are coherent, not a factual/coding qualification. Bare-text V2 empty completion is excluded. The 8K warmup was stopped to prioritize Vision; no 8K result is credited. Both owned Qwen driver and server exited. Evidence: `/var/tmp/atlas-prefill-census-qwen38-c2-chatml-20260905T0512Z/partial-review.json`; harness `/var/tmp/atlas-prefill-census-v3.U2vnFc/census`, SHA256 `bd52b54c9ccbd69e7ddedb9ee2f95167c89c734df5cd50e732f5460e8834802b`.

Flash-Next C3 exact canary passed; 256 tokens completed one warm + five measured, TTFT median 5924.344 ms, outputs byte-identical. At 2048 tokens, warm plus two measured completed: 48954.999/48996.712 ms, output hashes identical. Remaining repeats stopped for V11 correction; dropped trial3 response excluded. Short-run decode median 42.25 tok/s at 256 and individual 34.043/34.027 at 2048, not a standalone decode/long-context-quality qualification. Evidence `/var/tmp/atlas-prefill-census-flash-next-c3-20260905T0526Z/partial-review-v2.json` supersedes its one-sample review. Both owned driver and server exited; server graceful drain completed with exit0. Only the existing NVFP4-Offload checkpoint was served, not Mia mixed-MXFP8 donor weights.

Census V2: `/var/tmp/atlas-prefill-census-v2.eSOEED/census`, SHA256 `650bb182b785f03bf68bbe7871403c516b9f5a9835d6ad11ee2878db7c5016b1`. It explicitly disables canary thinking and allows `--bins 256,2048` for slow targets; unmeasured bins are reported. Use frozen per-model binary/config/argv/environment/PID/port provenance, no concurrent builds, and fresh root claims in `/home/flocka/atlas/qwen38/TEAM_INBOX.md` before every GPU/server lane.
