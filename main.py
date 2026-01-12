import os
import json
import re
import glob
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避
if sys.platform == 'win32':
    import os
    # 環境変数でUTF-8を強制（最初に設定）
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # コンソールのコードページをUTF-8に設定（PowerShell/CMD）
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True, check=False)
    except:
        pass
    
    # sys.stdoutのエンコーディング設定
    if hasattr(sys.stdout, 'buffer'):
        try:
            # 既にラップされている場合はスキップ
            if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

from model_definitions import initializeLLM, promptLLM, constructPrompt
from prompts import (multi_dim_prompt, NodeListSchema, type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema,
                     business_type_cls_system_instruction, business_type_cls_main_prompt, BusinessTypeClsSchema)
from taxonomy import Node, DAG
from expansion import expandNodeWidth, expandNodeDepth, shouldExpandWidthWithLLM
from paper import Paper
from utils import clean_json_string
from tools.markdown_utils import extract_metadata_from_markdown


def construct_dataset(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    split = 'train'
    
    if args.dataset == 'casestudy':
        # casestudyフォルダからMarkdownファイルを読み込む
        casestudy_dir = 'assets/casestudy'
        md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
        md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX')]
        md_files.sort()
        
        internal_collection = {}
        company_info_cache = {}
        
        with open(os.path.join(args.data_dir, 'internal.txt'), 'w', encoding='utf-8') as i:
            internal_count = 0
            id = 0
            
            for md_file in tqdm(md_files, desc="Loading casestudy files", ncols=100, ascii=False):
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # タイトルを抽出（最初の#見出し）
                    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                    if title_match:
                        title = title_match.group(1).strip()
                    else:
                        # ファイル名から抽出
                        basename = os.path.basename(md_file)
                        title = re.sub(r'^\d+_', '', basename).replace('.md', '')
                    
                    # 本文（abstract相当）は全内容を使用
                    abstract = content
                    
                    # 会社名を抽出（YAMLフロントマターからのみ、必須）
                    metadata = extract_metadata_from_markdown(content)
                    if not metadata:
                        raise ValueError(f"エラー: {md_file} にYAMLフロントマターがありません。YAMLフロントマターを追加してください。")
                    
                    company_name = metadata.get('company_name')
                    if not company_name:
                        raise ValueError(f"エラー: {md_file} のYAMLフロントマターに 'company_name' が設定されていません。")
                    
                    # 会社情報を抽出（YAMLフロントマターからのみ）
                    company_info = {}
                    if metadata.get('company_industry'):
                        company_info['industry'] = metadata['company_industry']
                    if metadata.get('company_revenue_size'):
                        company_info['revenue_size'] = metadata['company_revenue_size']
                    
                    if company_info:
                        company_info_cache[company_name] = company_info
                    
                    # Paperオブジェクトを作成
                    paper = Paper(id, title, abstract, label_opts=args.dimensions, internal=True)
                    paper.company_name = company_name
                    paper.company_industry = company_info.get('industry')
                    paper.company_revenue_size = company_info.get('revenue_size')
                    
                    temp_dict = {"Title": title, "Abstract": abstract, "Company": company_name}
                    if company_info.get('industry'):
                        temp_dict["Industry"] = company_info.get('industry')
                    if company_info.get('revenue_size'):
                        temp_dict["RevenueSize"] = company_info.get('revenue_size')
                    
                    formatted_dict = json.dumps(temp_dict, ensure_ascii=False)
                    i.write(f'{formatted_dict}\n')
                    internal_collection[id] = paper
                    internal_count += 1
                    id += 1
                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
                    continue
            
            print("Total # of Papers: ", internal_count)
        
        # 会社情報をJSONファイルに自動生成
        if company_info_cache:
            cache_file = os.path.join(args.data_dir, 'company_info.json')
            # 既存のファイルがあれば読み込んでマージ（手動入力の情報を保持）
            existing_info = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        existing_info = json.load(f)
                except:
                    pass
            
            # Markdownから抽出した情報で更新（既存の情報は保持）
            merged_info = existing_info.copy()
            for company_name, info in company_info_cache.items():
                if company_name not in merged_info or not merged_info[company_name].get('industry') or not merged_info[company_name].get('revenue_size'):
                    merged_info[company_name] = info
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(merged_info, f, ensure_ascii=False, indent=2)
        
        return internal_collection, internal_count
    
    # その他のデータセットは未対応
    raise ValueError(f"未対応のデータセット: {args.dataset}。'casestudy'のみ対応しています。")

def initialize_DAG(args):
    ## we want to make this a directed acyclic graph (DAG) so maintain a list of the nodes
    roots = {}
    id2node = {}
    label2node = {}
    idx = 0

    for dim in args.dimensions:
        mod_topic = args.topic.replace(' ', '_').lower()
        mod_full_topic = args.topic.replace(' ', '_').lower() + f"_{dim}"
        root = Node(
                id=idx,
                label=mod_topic,
                dimension=dim
            )
        roots[dim] = root
        id2node[idx] = root
        label2node[mod_full_topic] = root
        idx += 1

    queue = deque([node for id, node in id2node.items()])

    # if taking long, you can probably parallelize this between the different taxonomies (expand by level)
    while queue:
        curr_node = queue.popleft()
        label = curr_node.label
        dim = curr_node.dimension
        # expand
        system_instruction, main_prompt, json_output_format = multi_dim_prompt(curr_node)
        prompts = [constructPrompt(args, system_instruction, main_prompt + "\n\n" + json_output_format)]
        outputs = promptLLM(args=args, prompts=prompts, schema=NodeListSchema, max_new_tokens=3000, json_mode=True, temperature=0.01, top_p=1.0)[0]
        outputs = json.loads(clean_json_string(outputs)) if "```" in outputs else json.loads(outputs.strip())
        outputs = outputs['root_topic'] if 'root_topic' in outputs else outputs[label]

        # add all children
        for key, value in outputs.items():
            mod_key = key.replace(' ', '_').lower()
            mod_full_key = mod_key + f"_{dim}"
            if mod_full_key not in label2node:
                child_node = Node(
                        id=len(id2node),
                        label=mod_key,
                        dimension=dim,
                        description=value['description'],
                        parents=[curr_node]
                    )
                curr_node.add_child(mod_key, child_node)
                id2node[child_node.id] = child_node
                label2node[mod_full_key] = child_node
                if child_node.level < args.init_levels:
                    queue.append(child_node)
            elif label2node[mod_full_key] in label2node[label + f"_{dim}"].get_ancestors():
                continue
            else:
                child_node = label2node[mod_full_key]
                curr_node.add_child(mod_key, child_node)
                child_node.add_parent(curr_node)

    return roots, id2node, label2node


def main(args):

    print("######## STEP 1: LOAD IN DATASET ########")

    internal_collection, internal_count = construct_dataset(args)
    
    print(f'Internal: {internal_count}')

    # データサイズに応じてmax_densityを調整
    if args.dataset == 'casestudy':
        # 小規模データセットの場合は、データ数の10-15%程度を閾値にする
        if internal_count < 200:
            original_max_density = args.max_density
            args.max_density = max(10, int(internal_count * 0.15))  # 97件なら約14件
            print(f"Adjusted max_density from {original_max_density} to {args.max_density} for small dataset ({internal_count} papers)")

    print("######## STEP 2: INITIALIZE DAG ########")
    args = initializeLLM(args)

    roots, id2node, label2node = initialize_DAG(args)

    for dim in args.dimensions:
        with open(f'{args.data_dir}/initial_taxo_{dim}.txt', 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                roots[dim].display(0, indent_multiplier=5)

    print("######## STEP 3: CLASSIFY PAPERS BY DIMENSION (TASK, METHOD, DATASET, EVAL, APPLICATION, etc.) ########")

    # args.llm = 'vllm'  # GPTを使用する場合はコメントアウト
    dags = {dim:DAG(root=root, dim=dim) for dim, root in roots.items()}

    # do for internal collection
    # ビジネス向けかどうかでスキーマとプロンプトを切り替え
    if args.dataset == 'casestudy':
        system_instruction = business_type_cls_system_instruction
        main_prompt_func = business_type_cls_main_prompt
        schema = BusinessTypeClsSchema
    else:
        system_instruction = type_cls_system_instruction
        main_prompt_func = type_cls_main_prompt
        schema = TypeClsSchema

    prompts = [constructPrompt(args, system_instruction, main_prompt_func(paper)) for paper in internal_collection.values()]
    outputs = promptLLM(args=args, prompts=prompts, schema=schema, max_new_tokens=500, json_mode=True, temperature=0.1, top_p=0.99)
    outputs = [json.loads(clean_json_string(c)) if "```" in c else json.loads(c.strip()) for c in outputs]

    for r in roots:
        roots[r].papers = {}
    type_dist = {dim:[] for dim in args.dimensions}
    for p_id, out in enumerate(outputs):
        paper = internal_collection[p_id]
        paper.labels = {}
        
        # 既存のIndustryとRevenueSizeタグがある場合は、業種と会社規模を自動的にTrueに設定
        if args.dataset == 'casestudy':
            if hasattr(paper, 'company_industry') and paper.company_industry:
                out['業種'] = True
            if hasattr(paper, 'company_revenue_size') and paper.company_revenue_size:
                out['会社規模'] = True
        
        for key, val in out.items():
            if val:
                type_dist[key].append(paper)
                paper.labels[key] = []
                roots[key].papers[p_id] = paper
    
    print(str({k:len(v) for k,v in type_dist.items()}))


    # for each node, classify its papers for the children or perform depth expansion
    print("######## STEP 4: ITERATIVELY CLASSIFY & EXPAND ########")

    visited = set()
    queue = deque([roots[r] for r in roots])
    
    # 途中経過保存用のカウンター
    iteration_count = 0
    save_interval = 10  # 10ノード処理ごとに保存
    
    # 既存データがあるディメンションは幅方向展開をスキップ
    skip_width_expansion_dims = set()
    if args.dataset == 'casestudy':
        # 業種と会社規模は既存データがあるため、幅方向展開をスキップ
        skip_width_expansion_dims = {'業種', '会社規模'}
    
    # 会社規模は深さ方向展開もスキップ
    skip_depth_expansion_dims = set()
    if args.dataset == 'casestudy':
        skip_depth_expansion_dims = {'会社規模'}

    while queue:
        curr_node = queue.popleft()
        iteration_count += 1
        print(f'VISITING {curr_node.label} ({curr_node.dimension}) AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed (業種と会社規模は条件付きでスキップ)
            if curr_node.dimension not in skip_width_expansion_dims:
                new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
                print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')
            else:
                # 未分類論文の数を確認
                unlabeled_count = sum(1 for idx in curr_node.papers 
                                     if not any(idx in c.papers for c in curr_node.children.values()))
                # 未分類論文が多い場合のみ幅方向展開を実行
                if unlabeled_count > args.max_density:
                    print(f'(WIDTH EXPANSION) {curr_node.label} ({curr_node.dimension}) has {unlabeled_count} unlabeled papers, executing width expansion despite skip flag')
                    new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
                else:
                    new_sibs = []
                    print(f'(WIDTH EXPANSION SKIPPED) {curr_node.label} ({curr_node.dimension}) has {unlabeled_count} unlabeled papers (threshold: {args.max_density}), skipping width expansion')
            
            # 業務課題などの特定ディメンションでは、未分類論文が少なくてもLLMに判断を委ねる
            llm_judgment_dims = {'業務課題'}  # LLM判断を適用するディメンション
            if curr_node.dimension in llm_judgment_dims and len(new_sibs) == 0:
                # 未分類論文が少ない場合でも、LLMに判断を委ねる
                should_expand, reasoning = shouldExpandWidthWithLLM(args, curr_node, id2node, label2node)
                if should_expand:
                    print(f'(WIDTH EXPANSION - LLM JUDGMENT) {curr_node.label} ({curr_node.dimension}): {reasoning}')
                    new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
                    print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')
                else:
                    print(f'(WIDTH EXPANSION SKIPPED - LLM JUDGMENT) {curr_node.label} ({curr_node.dimension}): {reasoning}')

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                curr_node.classify_node(args, label2node, visited)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # 会社規模の場合は深さ方向展開をスキップ
            if curr_node.dimension in skip_depth_expansion_dims:
                print(f'(DEPTH EXPANSION SKIPPED) {curr_node.label} ({curr_node.dimension}) - skipping depth expansion')
                continue
            # no children -> perform depth expansion
            new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            # args.llm = 'vllm'  # GPTを使用する場合はコメントアウト
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label} ({curr_node.dimension}) are: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
        
        # 途中経過を定期的に保存
        if iteration_count % save_interval == 0:
            print(f"######## CHECKPOINT: Saving intermediate taxonomy at iteration {iteration_count} ########")
            for dim in args.dimensions:
                with open(f'{args.data_dir}/intermediate_taxo_{dim}_iter{iteration_count}.txt', 'w', encoding='utf-8') as f:
                    with redirect_stdout(f):
                        roots[dim].display(0, indent_multiplier=5)
    
    print("######## STEP 5: SAVE THE TAXONOMY ########")
    for dim in args.dimensions:
        with open(f'{args.data_dir}/final_taxo_{dim}.txt', 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=5)

        with open(f'{args.data_dir}/final_taxo_{dim}.json', 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='ビジネスソリューション導入事例')
    parser.add_argument('--dataset', type=str, default='casestudy')
    parser.add_argument('--llm', type=str, default='gpt', choices=['gpt', 'vllm', 'local'], 
                        help='使用するLLM: gpt (OpenAI API), vllm (vLLM), local (LM Studio/Ollama)')
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--init_levels', type=int, default=1)
    parser.add_argument('--max_density', type=int, default=40)
    parser.add_argument('--local_api_url', type=str, default=None,
                        help='ローカルAPIのベースURL (例: http://localhost:1234/v1 for LM Studio)')
    parser.add_argument('--local_api_key', type=str, default=None,
                        help='ローカルAPIのキー (LM Studio/Ollamaでは任意の文字列でOK)')
    parser.add_argument('--local_model_name', type=str, default=None,
                        help='ローカルモデル名 (例: deepseek-r1-distill-llama-70b-iq3-xs)')
    args = parser.parse_args()

    # casestudyの場合はビジネス向けディメンションを使用
    if args.dataset == 'casestudy':
        args.dimensions = ["業種", "業務領域", "導入効果", "技術領域", "導入形態", "業務課題", "会社規模"]
    else:
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]

    args.data_dir = f"datasets/{args.dataset.lower().replace(' ', '_')}"
    args.internal = f"{args.dataset}.txt"

    main(args)