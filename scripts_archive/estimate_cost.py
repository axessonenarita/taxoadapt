#!/usr/bin/env python3
"""
TaxoAdapt実行時のコスト概算スクリプト
"""
import json

# データセット情報
NUM_PAPERS = 96  # casestudyファイル数
NUM_DIMENSIONS = 7  # 業種、業務領域、導入効果、技術領域、導入形態、業務課題、会社規模
MAX_DEPTH = 2  # デフォルト
INIT_LEVELS = 1  # デフォルト
MAX_DENSITY = 40  # デフォルト

# GPT-4o-miniの価格（2024年12月時点）
# https://openai.com/api/pricing/
GPT4O_MINI_INPUT_PRICE_PER_1M = 0.15  # $/1M tokens
GPT4O_MINI_OUTPUT_PRICE_PER_1M = 0.60  # $/1M tokens

# 概算トークン数（プロンプトの長さから推定）
# 初期タクソノミー構築: 約2000 tokens (input) + 500 tokens (output)
# 論文分類: 約3000 tokens (input) + 100 tokens (output) per paper
# ノード分類: 約4000 tokens (input) + 200 tokens (output) per paper
# 拡張: 約3000 tokens (input) + 500 tokens (output) per expansion

def estimate_cost():
    """コストを概算"""
    
    # STEP 2: 初期タクソノミー構築
    # 各ディメンションごとに初期レベルを構築
    init_calls = NUM_DIMENSIONS * INIT_LEVELS
    init_input_tokens = init_calls * 2000
    init_output_tokens = init_calls * 500
    
    # STEP 3: 論文のディメンション分類
    # 全論文を7ディメンションで分類（1回の呼び出しで7つのbooleanを返す）
    classification_calls = NUM_PAPERS
    classification_input_tokens = classification_calls * 3000
    classification_output_tokens = classification_calls * 100
    
    # STEP 4: 反復的分類と拡張
    # これはタクソノミーの構造に依存するが、概算として：
    # - 各ディメンションで平均5-10ノードが生成されると仮定
    # - 各ノードで平均20論文が分類されると仮定
    # - 拡張は平均2-3回/ディメンションと仮定
    
    avg_nodes_per_dim = 7  # 保守的な見積もり
    avg_papers_per_node = 15  # 96論文 / 7ノード程度
    avg_expansions_per_dim = 3
    
    # ノード分類
    node_classification_calls = NUM_DIMENSIONS * avg_nodes_per_dim
    node_classification_input_tokens = node_classification_calls * avg_papers_per_node * 4000
    node_classification_output_tokens = node_classification_calls * avg_papers_per_node * 200
    
    # 拡張（幅・深さ）
    expansion_calls = NUM_DIMENSIONS * avg_expansions_per_dim
    expansion_input_tokens = expansion_calls * 3000
    expansion_output_tokens = expansion_calls * 500
    
    # 合計
    total_input_tokens = (init_input_tokens + 
                         classification_input_tokens + 
                         node_classification_input_tokens + 
                         expansion_input_tokens)
    
    total_output_tokens = (init_output_tokens + 
                           classification_output_tokens + 
                           node_classification_output_tokens + 
                           expansion_output_tokens)
    
    # コスト計算
    input_cost = (total_input_tokens / 1_000_000) * GPT4O_MINI_INPUT_PRICE_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * GPT4O_MINI_OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost
    
    # 結果を表示
    print("=" * 60)
    print("TaxoAdapt 実行コスト概算")
    print("=" * 60)
    print(f"\nデータセット情報:")
    print(f"  論文数: {NUM_PAPERS}")
    print(f"  ディメンション数: {NUM_DIMENSIONS}")
    print(f"  最大深度: {MAX_DEPTH}")
    print(f"  初期レベル: {INIT_LEVELS}")
    print(f"  最大密度: {MAX_DENSITY}")
    
    print(f"\nLLM呼び出し回数（概算）:")
    print(f"  初期タクソノミー構築: {init_calls}回")
    print(f"  論文分類: {classification_calls}回")
    print(f"  ノード分類: {node_classification_calls}回（推定）")
    print(f"  拡張: {expansion_calls}回（推定）")
    print(f"  合計: {init_calls + classification_calls + node_classification_calls + expansion_calls}回（推定）")
    
    print(f"\nトークン数（概算）:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  合計: {total_input_tokens + total_output_tokens:,}")
    
    print(f"\nコスト（GPT-4o-mini使用時）:")
    print(f"  Input: ${input_cost:.2f}")
    print(f"  Output: ${output_cost:.2f}")
    print(f"  合計: ${total_cost:.2f} (約{total_cost * 150:.0f}円)")
    
    print(f"\n注意事項:")
    print(f"  - この概算は保守的な見積もりです")
    print(f"  - 実際のコストはタクソノミーの構造に大きく依存します")
    print(f"  - 拡張が多くなるとコストも増加します")
    print(f"  - vLLMを使用する部分は無料です（ローカル実行）")
    print(f"  - デフォルトでは初期化のみGPT、その他はvLLMを使用します")
    
    # vLLM使用時のコスト（ほぼ0）
    print(f"\n実際のコスト（vLLM使用時）:")
    print(f"  - STEP 2（初期化）のみGPT使用: 約${(init_input_tokens/1_000_000)*GPT4O_MINI_INPUT_PRICE_PER_1M + (init_output_tokens/1_000_000)*GPT4O_MINI_OUTPUT_PRICE_PER_1M:.2f}")
    print(f"  - その他はvLLM（ローカル）: 無料")
    print(f"  - 合計: 約${(init_input_tokens/1_000_000)*GPT4O_MINI_INPUT_PRICE_PER_1M + (init_output_tokens/1_000_000)*GPT4O_MINI_OUTPUT_PRICE_PER_1M:.2f} (約{(init_input_tokens/1_000_000)*GPT4O_MINI_INPUT_PRICE_PER_1M*150 + (init_output_tokens/1_000_000)*GPT4O_MINI_OUTPUT_PRICE_PER_1M*150:.0f}円)")
    
    print("=" * 60)

if __name__ == '__main__':
    estimate_cost()
