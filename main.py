import os
import json
import re
import glob
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm

from model_definitions import initializeLLM, promptLLM, constructPrompt
from prompts import (multi_dim_prompt, NodeListSchema, type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema,
                     business_type_cls_system_instruction, business_type_cls_main_prompt, BusinessTypeClsSchema)
from taxonomy import Node, DAG
from datasets import load_dataset
from expansion import expandNodeWidth, expandNodeDepth
from paper import Paper
from utils import clean_json_string

def extract_company_name(content):
    """Markdownファイルから会社名を抽出"""
    # パターン1: 「### 会社名」形式（基本情報セクション内）
    pattern1 = r'###\s+([^#\n]+(?:株式会社|有限会社|合資会社|合名会社|合同会社|一般社団法人|財団法人|協同組合|組合)[^\n]*)'
    match1 = re.search(pattern1, content)
    if match1:
        company_name = match1.group(1).strip()
        # 「様」を除去
        company_name = company_name.replace('様', '').strip()
        return company_name
    
    # パターン2: 「会社名様」形式（タイトルの直後）
    pattern2 = r'^#\s+[^\n]+\n+\n+([^#\n]+(?:株式会社|有限会社|合資会社|合名会社|合同会社|一般社団法人|財団法人|協同組合|組合)[^\n]*様?)'
    match2 = re.search(pattern2, content, re.MULTILINE)
    if match2:
        company_name = match2.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        return company_name
    
    return None

def extract_metadata_from_markdown(content):
    """MarkdownファイルのYAMLフロントマターからメタデータを抽出"""
    # YAMLフロントマターのパターン（---で囲まれた部分）
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    frontmatter_match = re.search(frontmatter_pattern, content, re.DOTALL | re.MULTILINE)
    
    if not frontmatter_match:
        return None
    
    frontmatter_text = frontmatter_match.group(1)
    metadata = {}
    
    # YAML形式のパース（簡易版）
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # key: value 形式をパース
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            # クォートを除去
            value = value.strip('"\'')
            metadata[key] = value
    
    return metadata if metadata else None

def extract_company_info_from_markdown(content, company_name):
    """Markdownファイルから会社情報を抽出（YAMLフロントマター優先、基本情報セクションはフォールバック）"""
    if not company_name:
        return None
    
    # まずYAMLフロントマターから抽出を試みる
    metadata = extract_metadata_from_markdown(content)
    if metadata:
        company_info = {}
        if 'company_industry' in metadata or 'industry' in metadata:
            company_info['industry'] = metadata.get('company_industry') or metadata.get('industry')
        if 'company_revenue_size' in metadata or 'revenue_size' in metadata:
            company_info['revenue_size'] = metadata.get('company_revenue_size') or metadata.get('revenue_size')
        
        if company_info.get('industry') or company_info.get('revenue_size'):
            return company_info
    
    # フロントマターに情報がない場合は、基本情報セクションから抽出（フォールバック）
    basic_info_pattern = r'##\s+基本情報\s+(.*?)(?=##|\Z)'
    basic_info_match = re.search(basic_info_pattern, content, re.DOTALL)
    
    if not basic_info_match:
        return None
    
    basic_info_section = basic_info_match.group(1)
    
    # 会社名のセクションを抽出
    company_section_pattern = rf'###\s+{re.escape(company_name)}.*?(?=###|\Z)'
    company_section_match = re.search(company_section_pattern, basic_info_section, re.DOTALL)
    
    if not company_section_match:
        return None
    
    company_section = company_section_match.group(0)
    
    # 業種を抽出
    industry_pattern = r'業種[：:]\s*([^\n]+)'
    industry_match = re.search(industry_pattern, company_section)
    industry = industry_match.group(1).strip() if industry_match else None
    
    # 売上規模を抽出
    revenue_pattern = r'売上規模[：:]\s*([^\n]+)'
    revenue_match = re.search(revenue_pattern, company_section)
    revenue_size = revenue_match.group(1).strip() if revenue_match else None
    
    if industry or revenue_size:
        return {
            'industry': industry,
            'revenue_size': revenue_size
        }
    
    return None

def investigate_company_info(company_name, manual_info_file='datasets/casestudy/company_info.json'):
    """手動入力ファイルから会社の業種と売上規模を読み込む"""
    import json
    
    # 手動入力ファイルを読み込み
    if os.path.exists(manual_info_file):
        try:
            with open(manual_info_file, 'r', encoding='utf-8') as f:
                manual_info = json.load(f)
            
            # 会社名で検索（完全一致または部分一致）
            if company_name in manual_info:
                return manual_info[company_name]
            
            # 部分一致で検索（「株式会社」などの表記揺れに対応）
            for key, value in manual_info.items():
                if company_name in key or key in company_name:
                    return value
        except Exception as e:
            print(f"Warning: Failed to load company info file: {e}")
    
    # 見つからない場合はNoneを返す
    return {
        'industry': None,
        'revenue_size': None
    }

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
            
            for md_file in tqdm(md_files, desc="Loading casestudy files"):
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
                    
                    # 会社名を抽出
                    company_name = extract_company_name(content)
                    
                    # 会社情報を抽出（Markdownファイルから優先、なければ手動入力ファイルから）
                    company_info = None
                    if company_name:
                        # まずMarkdownファイルから抽出を試みる
                        company_info = extract_company_info_from_markdown(content, company_name)
                        
                        # Markdownファイルに情報がない場合は、手動入力ファイルから読み込む
                        if not company_info or (not company_info.get('industry') and not company_info.get('revenue_size')):
                            manual_info = investigate_company_info(company_name, 
                                                                   manual_info_file=os.path.join(args.data_dir, 'company_info.json'))
                            # 手動入力ファイルの情報で上書き（Markdownの情報があれば優先）
                            if manual_info:
                                if not company_info:
                                    company_info = {}
                                if not company_info.get('industry') and manual_info.get('industry'):
                                    company_info['industry'] = manual_info['industry']
                                if not company_info.get('revenue_size') and manual_info.get('revenue_size'):
                                    company_info['revenue_size'] = manual_info['revenue_size']
                        
                        if company_info:
                            company_info_cache[company_name] = company_info
                    
                    # Paperオブジェクトを作成
                    paper = Paper(id, title, abstract, label_opts=args.dimensions, internal=True)
                    if company_info:
                        paper.company_name = company_name
                        paper.company_industry = company_info.get('industry')
                        paper.company_revenue_size = company_info.get('revenue_size')
                    
                    temp_dict = {"Title": title, "Abstract": abstract}
                    if company_name:
                        temp_dict["Company"] = company_name
                    if company_info:
                        temp_dict["Industry"] = company_info.get('industry')
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
    
    # 既存のデータセット処理
    if args.dataset == 'emnlp_2024':
        ds = load_dataset("EMNLP/EMNLP2024-papers")
    elif args.dataset == 'emnlp_2022':
        ds = load_dataset("TimSchopf/nlp_taxonomy_data")
        split = 'test'
    elif args.dataset == 'cvpr_2024':
        ds = load_dataset("DeepNLP/CVPR-2024-Accepted-Papers")
    elif args.dataset == 'cvpr_2020':
        ds = load_dataset("DeepNLP/CVPR-2020-Accepted-Papers")
    elif args.dataset == 'iclr_2024':
        ds = load_dataset("DeepNLP/ICLR-2024-Accepted-Papers")
    elif args.dataset == 'iclr_2021':
        ds = load_dataset("DeepNLP/ICLR-2021-Accepted-Papers")
    elif args.dataset == 'icra_2024':
        ds = load_dataset("DeepNLP/ICRA-2024-Accepted-Papers")
    else:
        ds = load_dataset("DeepNLP/ICRA-2020-Accepted-Papers")
    
    
    internal_collection = {}

    with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as i:
        internal_count = 0
        id = 0
        for p in tqdm(ds[split]):
            if ('title' not in p) and ('abstract' not in p):
                continue
            
            temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
            formatted_dict = json.dumps(temp_dict)
            i.write(f'{formatted_dict}\n')
            internal_collection[id] = Paper(id, p['title'], p['abstract'], label_opts=args.dimensions, internal=True)
            internal_count += 1
            id += 1
        print("Total # of Papers: ", internal_count)
    
    return internal_collection, internal_count

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

    print("######## STEP 2: INITIALIZE DAG ########")
    args = initializeLLM(args)

    roots, id2node, label2node = initialize_DAG(args)

    for dim in args.dimensions:
        with open(f'{args.data_dir}/initial_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                roots[dim].display(0, indent_multiplier=5)

    print("######## STEP 3: CLASSIFY PAPERS BY DIMENSION (TASK, METHOD, DATASET, EVAL, APPLICATION, etc.) ########")

    args.llm = 'vllm'
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
        internal_collection[p_id].labels = {}
        for key, val in out.items():
            if val:
                type_dist[key].append(internal_collection[p_id])
                internal_collection[p_id].labels[key] = []
                roots[key].papers[p_id] = internal_collection[p_id]
    
    print(str({k:len(v) for k,v in type_dist.items()}))


    # for each node, classify its papers for the children or perform depth expansion
    print("######## STEP 4: ITERATIVELY CLASSIFY & EXPAND ########")

    visited = set()
    queue = deque([roots[r] for r in roots])

    while queue:
        curr_node = queue.popleft()
        print(f'VISITING {curr_node.label} ({curr_node.dimension}) AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed
            new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
            print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                curr_node.classify_node(args, label2node, visited)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # no children -> perform depth expansion
            new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            args.llm = 'vllm'
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label} ({curr_node.dimension}) are: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
    
    print("######## STEP 5: SAVE THE TAXONOMY ########")
    for dim in args.dimensions:
        with open(f'{args.data_dir}/final_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=5)

        with open(f'{args.data_dir}/final_taxo_{dim}.json', 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='ビジネスソリューション導入事例')
    parser.add_argument('--dataset', type=str, default='casestudy')
    parser.add_argument('--llm', type=str, default='gpt')
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--init_levels', type=int, default=1)
    parser.add_argument('--max_density', type=int, default=40)
    args = parser.parse_args()

    # casestudyの場合はビジネス向けディメンションを使用
    if args.dataset == 'casestudy':
        args.dimensions = ["業種", "業務領域", "導入効果", "技術領域", "導入形態", "業務課題", "会社規模"]
    else:
        args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]

    args.data_dir = f"datasets/{args.dataset.lower().replace(' ', '_')}"
    args.internal = f"{args.dataset}.txt"

    main(args)