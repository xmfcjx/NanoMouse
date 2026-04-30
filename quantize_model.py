"""
模型量化脚本
生成 GGUF 和 GPTQ 格式的量化模型
"""
import os
import torch

MODEL_PATH = "/mnt/d/Desktop/NanoChat-Lab/models/Qwen2.5-1.5B"
OUTPUT_DIR = "/mnt/d/Desktop/NanoChat-Lab/models"


def quantize_gptq():
    """使用 GPTQ 量化模型"""
    print("=" * 60)
    print("【GPTQ 量化】")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    except ImportError:
        print("❌ 请先安装: pip install transformers>=4.33.0")
        return
    
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    except ImportError:
        print("❌ 请先安装: pip install auto-gptq")
        return
    
    output_path = os.path.join(OUTPUT_DIR, "qwen2.5-1.5b-gptq-int4")
    
    print("加载原始模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    calibration_data = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习和改进。",
        "深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的工作方式。",
        "自然语言处理是人工智能的重要应用领域，专注于计算机与人类语言之间的交互。",
        "计算机视觉是让机器能够理解和解释视觉信息的技术。",
    ]
    
    print("准备校准数据...")
    examples = [tokenizer(d) for d in calibration_data]
    
    print("开始 GPTQ 量化...")
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )
    
    model = AutoGPTQForCausalLM.from_pretrained(
        MODEL_PATH,
        quantize_config,
        trust_remote_code=True,
    )
    
    model.quantize(examples)
    
    print(f"保存量化模型到: {output_path}")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ GPTQ 量化完成!")


def convert_to_gguf():
    """转换为 GGUF 格式"""
    print("\n" + "=" * 60)
    print("【GGUF 转换】")
    print("=" * 60)
    
    print("""
GGUF 转换需要使用 llama.cpp 工具，步骤如下：

1. 克隆 llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

2. 安装依赖:
   pip install -r requirements.txt

3. 转换模型:
   python convert-hf-to-gguf.py {MODEL_PATH} --outfile qwen2.5-1.5b-f16.gguf --outtype f16

4. 量化为 INT4:
   ./llama-quantize qwen2.5-1.5b-f16.gguf qwen2.5-1.5b-q4_k_m.gguf Q4_K_M

5. 移动到项目目录:
   mv qwen2.5-1.5b-q4_k_m.gguf {OUTPUT_DIR}/

或者直接下载已量化的 GGUF 模型:
   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \\
     qwen2.5-1.5b-instruct-q4_k_m.gguf \\
     --local-dir {OUTPUT_DIR}
""".format(MODEL_PATH=MODEL_PATH, OUTPUT_DIR=OUTPUT_DIR))


def download_gguf():
    """下载预量化的 GGUF 模型"""
    print("=" * 60)
    print("【下载 GGUF 模型】")
    print("=" * 60)
    
    import os
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ 请先安装: pip install huggingface_hub")
        return
    
    print("下载 Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)...")
    print("使用镜像: https://hf-mirror.com")
    
    try:
        local_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
            local_dir=OUTPUT_DIR,
            endpoint="https://hf-mirror.com",
        )
        print(f"✅ 下载完成: {local_path}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n手动下载方法:")
        print("1. 访问: https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF")
        print("2. 下载: qwen2.5-1.5b-instruct-q4_k_m.gguf")
        print(f"3. 放到: {OUTPUT_DIR}/")


def main():
    print("\n" + "=" * 60)
    print("  模型量化工具")
    print("=" * 60)
    
    print("""
选择操作:
1. GPTQ 量化 (需要 auto-gptq)
2. GGUF 转换说明
3. 下载预量化 GGUF 模型 (推荐)
4. 全部执行
0. 退出
""")
    
    choice = input("请选择 [0-4]: ").strip()
    
    if choice == "1":
        quantize_gptq()
    elif choice == "2":
        convert_to_gguf()
    elif choice == "3":
        download_gguf()
    elif choice == "4":
        quantize_gptq()
        convert_to_gguf()
        download_gguf()
    else:
        print("退出")


if __name__ == "__main__":
    main()
