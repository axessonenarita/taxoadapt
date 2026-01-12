#!/usr/bin/env python3
"""
TaxoAdapt実行時のコスト概算ツール
"""
import json

# GPT-4o-miniの価格（2024年12月時点）
GPT4O_MINI_INPUT_PRICE_PER_1M = 0.15  # $/1M tokens
GPT4O_MINI_OUTPUT_PRICE_PER_1M = 0.60  # $/1M tokens


def estimate_cost(num_papers=96, num_dimensions=7, max_depth=2, init_levels=1, max_density=40):
    """コストを概算"""
    
    # STEP 2: 初期タクソノミー構築
    init_calls = num_dimensions * init_levels
    init_input_tokens = init_calls * 2000
    init_output_tokens = init_calls * 500
    
    # STEP 3: 論文のディメンション分類
    classification_calls = num_papers
    classification_input_tokens = classification_calls * 3000
    classification_output_tokens = classification_calls * 100
    
    # STEP 4: 反復的分類と拡張（概算）
    avg_nodes_per_dim = 7
    avg_papers_per_node = 15
    avg_expansions_per_dim = 3
    
    # ノード分類
    node_classification_calls = num_dimensions * avg_nodes_per_dim
    node_classification_input_tokens = node_classification_calls * avg_papers_per_node * 4000
    node_classification_output_tokens = node_classification_calls * avg_papers_per_node * 200
    
    # 拡張（幅・深さ）
    expansion_calls = num_dimensions * avg_expansions_per_dim
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
    
    return {
        'num_papers': num_papers,
        'num_dimensions': num_dimensions,
        'max_depth': max_depth,
        'init_levels': init_levels,
        'max_density': max_density,
        'calls': {
            'init': init_calls,
            'classification': classification_calls,
            'node_classification': node_classification_calls,
            'expansion': expansion_calls,
            'total': init_calls + classification_calls + node_classification_calls + expansion_calls
        },
        'tokens': {
            'input': total_input_tokens,
            'output': total_output_tokens,
            'total': total_input_tokens + total_output_tokens
        },
        'cost': {
            'input': input_cost,
            'output': output_cost,
            'total': total_cost,
            'total_jpy': total_cost * 150  # 1ドル=150円で換算
        }
    }


def print_estimate(result):
    """概算結果を表示"""
    print("=" * 60)
    print("TaxoAdapt 実行コスト概算")
    print("=" * 60)
    print(f"\nデータセット情報:")
    print(f"  論文数: {result['num_papers']}")
    print(f"  ディメンション数: {result['num_dimensions']}")
    print(f"  最大深度: {result['max_depth']}")
    print(f"  初期レベル: {result['init_levels']}")
    print(f"  最大密度: {result['max_density']}")
    
    print(f"\nLLM呼び出し回数（概算）:")
    print(f"  初期タクソノミー構築: {result['calls']['init']}回")
    print(f"  論文分類: {result['calls']['classification']}回")
    print(f"  ノード分類: {result['calls']['node_classification']}回（推定）")
    print(f"  拡張: {result['calls']['expansion']}回（推定）")
    print(f"  合計: {result['calls']['total']}回（推定）")
    
    print(f"\nトークン数（概算）:")
    print(f"  Input tokens: {result['tokens']['input']:,}")
    print(f"  Output tokens: {result['tokens']['output']:,}")
    print(f"  合計: {result['tokens']['total']:,}")
    
    print(f"\nコスト（GPT-4o-mini使用時）:")
    print(f"  Input: ${result['cost']['input']:.2f}")
    print(f"  Output: ${result['cost']['output']:.2f}")
    print(f"  合計: ${result['cost']['total']:.2f} (約{result['cost']['total_jpy']:.0f}円)")
    
    print(f"\n注意事項:")
    print(f"  - この概算は保守的な見積もりです")
    print(f"  - 実際のコストはタクソノミーの構造に大きく依存します")
    print(f"  - 拡張が多くなるとコストも増加します")
    print(f"  - vLLMを使用する部分は無料です（ローカル実行）")
    print(f"  - デフォルトでは初期化のみGPT、その他はvLLMを使用します")
    
    # vLLM使用時のコスト（ほぼ0）
    init_cost = (result['calls']['init'] * 2000 / 1_000_000) * GPT4O_MINI_INPUT_PRICE_PER_1M + \
                (result['calls']['init'] * 500 / 1_000_000) * GPT4O_MINI_OUTPUT_PRICE_PER_1M
    print(f"\n実際のコスト（vLLM使用時）:")
    print(f"  - STEP 2（初期化）のみGPT使用: 約${init_cost:.2f}")
    print(f"  - その他はvLLM（ローカル）: 無料")
    print(f"  - 合計: 約${init_cost:.2f} (約{init_cost * 150:.0f}円)")
    
    print("=" * 60)


if __name__ == '__main__':
    result = estimate_cost()
    print_estimate(result)
