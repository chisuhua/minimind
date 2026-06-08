import os
import pickle
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from model.tri_attention import TriAttentionScorer
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        gated_layers = list(args.gated_deltanet_layers) if args.gated_deltanet_layers is not None else ([0] if args.gated_deltanet else None)
        lightning_layers = list(args.lightning_indexer_layers) if args.lightning_indexer_layers is not None else ([0] if args.lightning_indexer else None)
        if args.qwen3_next:
            hybrid_pattern = ['d'] * 6 + ['a'] * 2
            lm_config = MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=len(hybrid_pattern),
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
                head_dim=128,
                partial_rope_dim=32,
                hybrid_pattern=hybrid_pattern,
            )
            from model.model_minimind_hybrid import HybridMiniMindForCausalLM
            model = HybridMiniMindForCausalLM(lm_config)
            print(f"Qwen3-Next hybrid model: {len(hybrid_pattern)} layers ({sum(1 for t in hybrid_pattern if t == 'd')} DeltaNet + {sum(1 for t in hybrid_pattern if t == 'a')} GatedAttention)")
        else:
            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling,
                use_lookahead=bool(args.lookahead_decoding),
                use_streaming_llm=bool(args.streaming_llm),
                use_mtp=bool(args.mtp),
                gated_deltanet_layers=gated_layers,
                lightning_indexer_layers=lightning_layers,
            ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        if args.qwen3_next:
            hybrid_pattern = ['d'] * 6 + ['a'] * 2
            model.config.hybrid_pattern = hybrid_pattern
            model.config.partial_rope_dim = 32
            print(f"Qwen3-Next hybrid config applied: {len(hybrid_pattern)} layers ({sum(1 for t in hybrid_pattern if t == 'd')} DeltaNet + {sum(1 for t in hybrid_pattern if t == 'a')} GatedAttention)")
        if args.lookahead_decoding:
            model.config.use_lookahead = True
        if args.gated_deltanet:
            from model.gated_deltanet import GatedDeltaNet
            for l, block in enumerate(model.model.layers):
                if l in (args.gated_deltanet_layers or [0]):
                    block.self_attn = GatedDeltaNet(model.config)
            print(f"Replaced layers {args.gated_deltanet_layers or [0]} with GatedDeltaNet")
        if args.lightning_indexer:
            from model.lightning_indexer import SparseAttentionWithIndexer
            for l, block in enumerate(model.model.layers):
                if l in (args.lightning_indexer_layers or [0]):
                    block.self_attn = SparseAttentionWithIndexer(model.config)
            print(f"Replaced layers {args.lightning_indexer_layers or [0]} with LightningIndexer")
        if args.nsa_sparse:
            from model.nsa import replace_attention_with_nsa
            replace_attention_with_nsa(model, model.config)
            print("Replaced all Attention layers with NSA (Native Sparse Attention)")
        if args.mhc_residual:
            from model.mhc import MHCBlock
            for i, block in enumerate(model.model.layers):
                mhc_block = MHCBlock(i, model.config)
                mhc_block.load_state_dict(block.state_dict(), strict=False)
                model.model.layers[i] = mhc_block
            print("Replaced all layers with MHCBlock (mHC residual)")
        if args.pre_alloc_kv:
            for block in model.model.layers:
                block.self_attn.pre_alloc_kv = True
                block.self_attn.k_buf = None
                block.self_attn.v_buf = None
    if args.tri_attention:
        scorer_path = args.tri_scorer_path or f'{args.save_dir}/tri_scorers.pkl'
        if os.path.exists(scorer_path):
            with open(scorer_path, 'rb') as f:
                scorers = pickle.load(f)
            for layer, scorer in zip(model.model.layers, scorers):
                layer.self_attn.tri_scorer = scorer
            print(f"Loaded TriAttention scorers from {scorer_path}")
        else:
            print(f"Warning: {scorer_path} not found, TriAttention disabled")
    if args.rt_purbo:
        calib_path = args.rt_purbo_calib_path or f'{args.save_dir}/rt_purbo_heads.json'
        if os.path.exists(calib_path):
            from model.rt_purbo import RetrievalHeadClassifier, LowDimIndexer, RTPurboAttention
            classifier = RetrievalHeadClassifier.load(calib_path, model=model)
            indexer = LowDimIndexer(head_dim=model.config.head_dim, low_dim=16).half().to(args.device)
            indexer_path = f'{args.save_dir}/rt_purbo_indexer.pth'
            if os.path.exists(indexer_path):
                indexer.load_state_dict(torch.load(indexer_path, map_location=args.device))
                print(f"Loaded RTPurbo indexer from {indexer_path}")
            else:
                print(f"Warning: {indexer_path} not found, using random indexer")
            rtpurbo = RTPurboAttention(
                model, classifier.retrieval_heads, indexer,
                sink=args.rt_purbo_sink, local_window=args.rt_purbo_window, top_p=args.rt_purbo_top_p,
            )
            rtpurbo.attach()
            print(f"RTPurbo enabled: {sum(1 for v in classifier.retrieval_heads.values() if v)} retrieval heads"
                  f" (sink={args.rt_purbo_sink}, window={args.rt_purbo_window}, top_p={args.rt_purbo_top_p})")
        else:
            print(f"Warning: {calib_path} not found, RTPurbo disabled")

    if args.streaming_llm:
        model.config.use_streaming_llm = True
        for block in model.model.layers:
            block.self_attn.use_streaming_llm = True
            block.self_attn.streaming_kv = None
        num_sink = getattr(model.config, 'num_sink_tokens', 4) or 4
        window = getattr(model.config, 'sliding_window', 4096) or 4096
        print(f"StreamingLLM enabled: num_sink={num_sink}, window={window}")
    if args.kv_quant == 'kivi_2bit':
        model.config.use_kivi = True
        for block in model.model.layers:
            block.self_attn.use_kivi = True
            block.self_attn.kivi_cache = None
        print(f"KIVI 2-bit KV cache quantization enabled (window={model.config.kivi_window})")
    if args.minference:
        pattern_path = args.minference_pattern_path or f'{args.save_dir}/minference_patterns.json'
        if os.path.exists(pattern_path):
            from model.minference import OfflineSearcher, OnlineIndexer
            searcher = OfflineSearcher.load(pattern_path, model=model)
            if args.minference_pattern != 'auto':
                num_layers = len(model.model.layers)
                num_heads = model.config.num_attention_heads
                for l in range(num_layers):
                    for h in range(num_heads):
                        searcher.patterns[(l, h)] = args.minference_pattern
            indexer = OnlineIndexer(
                model, searcher.patterns,
                sink=searcher.sink, window=searcher.window,
                block=searcher.block, top_k=searcher.top_k,
                last_q=searcher.last_q, slash_stride=searcher.slash_stride,
                vertical_top=searcher.vertical_top,
            )
            indexer.attach()
            print(f"Loaded MInference patterns from {pattern_path} ({len(searcher.patterns)} head assignments)")
        else:
            print(f"Warning: {pattern_path} not found, MInference disabled (falling back to standard attention)")
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--pld', default=False, action='store_true', help="启用Prompt Lookup Decoding加速推理")
    parser.add_argument('--mtp', default=False, action='store_true', help="启用MTP-as-Draft多token预测加速推理")
    parser.add_argument('--lookahead_decoding', default=0, type=int, help="启用Lookahead Decoding加速推理（0=否，1=是）")
    parser.add_argument('--pre_alloc_kv', default=False, action='store_true', help="启用预分配KV Cache（替代torch.cat）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--tri_attention', default=False, action='store_true', help="启用TriAttention稀疏注意力掩码")
    parser.add_argument('--tri_scorer_path', default=None, type=str, help="TriAttention scorer文件路径（默认: ./out/tri_scorers.pkl）")
    parser.add_argument('--minference', default=False, action='store_true', help="启用MInference 1.0稀疏注意力（A-shape / Vertical-Slash / Block-Sparse）")
    parser.add_argument('--minference_pattern_path', default=None, type=str, help="MInference 标定文件路径（默认: ./out/minference_patterns.json）")
    parser.add_argument('--minference_pattern', default='auto', type=str, help="MInference 模式（auto=A/VS/BS 标定, A, VS, BS）")
    parser.add_argument('--rt_purbo', default=False, action='store_true', help="启用RTPurbo head-wise稀疏注意力")
    parser.add_argument('--rt_purbo_calib_path', default=None, type=str, help="RTPurbo retrieval head标定文件路径")
    parser.add_argument('--rt_purbo_sink', default=4, type=int, help="RTPurbo sink token数量")
    parser.add_argument('--rt_purbo_window', default=8192, type=int, help="RTPurbo local window大小")
    parser.add_argument('--rt_purbo_top_p', default=0.9, type=float, help="RTPurbo top-p选择阈值")
    parser.add_argument('--streaming_llm', default=False, action='store_true', help="启用StreamingLLM固定KV缓存 (sink+sliding window)")
    parser.add_argument('--kv_quant', default=None, type=str, choices=[None, 'kivi_2bit'], help="KV cache量化方法（kivi_2bit=KIVI 2-bit量化）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--medusa', default=False, action='store_true', help="启用Medusa-1并行解码加速推理")
    parser.add_argument('--medusa_heads', default=3, type=int, help="Medusa heads数量")
    parser.add_argument('--gated_deltanet', default=False, action='store_true', help="启用Gated DeltaNet替换指定注意力层")
    parser.add_argument('--gated_deltanet_layers', default=None, type=int, nargs='+', help="替换为Gated DeltaNet的层索引（默认: 0）")
    parser.add_argument('--lightning_indexer', default=False, action='store_true', help="启用Lightning Indexer DSA稀疏注意力")
    parser.add_argument('--lightning_indexer_layers', default=None, type=int, nargs='+', help="替换为Lightning Indexer的层索引（默认: 0）")
    parser.add_argument('--mhc_residual', default=False, action='store_true', help="启用mHC Hyper-Connection残差结构")
    parser.add_argument('--dflash', default=False, action='store_true', help="启用DFlash Block Diffusion推测解码")
    parser.add_argument('--dflash_block_size', default=16, type=int, help="DFlash block大小")
    parser.add_argument('--ddtree', default=False, action='store_true', help="启用DDTree树形多路推测解码（隐含--dflash）")
    parser.add_argument('--dflash_draft', default=False, action='store_true', help="启用DFlash Block Diffusion加速推理（--dflash的别名）")
    parser.add_argument('--qwen3_next', default=False, action='store_true', help="启用Qwen3-Next混合架构 (6 GatedDeltaNet + 2 GatedAttention)")
    parser.add_argument('--nsa_sparse', default=False, action='store_true', help="启用NSA (Native Sparse Attention) 3-branch稀疏注意力")
    parser.add_argument('--diffusion_decode', default=False, action='store_true', help="启用扩散语言模型采样（dLM）")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generate_kwargs = dict(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1,
        )
        if args.pld:
            generate_kwargs["use_pld"] = True
        if args.mtp:
            generate_kwargs["use_mtp"] = True
        if args.medusa:
            from model.medusa_heads import MedusaHeads
            medusa_heads = MedusaHeads(model.config)
            moe_suffix = '_moe' if args.use_moe else ''
            medusa_path = f'./{args.save_dir}/medusa_{args.medusa_heads}_{args.hidden_size}{moe_suffix}.pth'
            medusa_heads.load_state_dict(torch.load(medusa_path, map_location=args.device))
            medusa_heads = medusa_heads.half().to(args.device)
            generate_kwargs["medusa_heads"] = medusa_heads
        if args.ddtree or args.dflash:
            from model.dflash import DFlashDraftModel, DFlashSpecDecoder
            model.config.dflash_block_size = args.dflash_block_size
            dflash_draft = DFlashDraftModel(model.config, block_size=args.dflash_block_size)
            moe_suffix = '_moe' if args.use_moe else ''
            dflash_path = f'./{args.save_dir}/dflash_{model.config.dflash_block_size}_{args.hidden_size}{moe_suffix}.pth'
            if os.path.exists(dflash_path):
                dflash_draft.load_state_dict(torch.load(dflash_path, map_location=args.device))
                dflash_draft = dflash_draft.half().to(args.device)
                generate_kwargs["dflash_decoder"] = DFlashSpecDecoder(dflash_draft, model)
                print(f"DFlash draft loaded from {dflash_path}")
                if args.ddtree:
                    from model.ddtree import DDTreeDecoder
                    ddtree_decoder = DDTreeDecoder(model, verify=True)
                    generate_kwargs["ddtree_decoder"] = ddtree_decoder
                    print("DDTree multi-path verification enabled")
            else:
                print(f"Warning: {dflash_path} not found, DFlash/DDTree disabled")
        generated_ids = model.generate(**generate_kwargs)
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()