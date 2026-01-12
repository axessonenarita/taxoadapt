import os
from tqdm import tqdm
import openai
from openai import OpenAI

# Windowsのコンソール出力のエンコーディング問題を回避
# main.pyで既に設定されているため、ここでは環境変数のみ設定
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# オプショナルなインポート（必要な場合のみ）
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    LLM = None
    SamplingParams = None
    GuidedDecodingParams = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

# .envファイルから環境変数を読み込む（オプション）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenvがインストールされていない場合はスキップ

openai_key = os.getenv('OPENAI_API_KEY')
local_api_url = os.getenv('LOCAL_API_URL', 'http://localhost:1234/v1')
local_api_key = os.getenv('LOCAL_API_KEY', 'lm-studio')
local_model_name = os.getenv('LOCAL_MODEL_NAME', 'qwen/qwen3-vl-30b')


# map each term in text to word_id
def get_vocab_idx(split_text: str, tok_lens):

	vocab_idx = {}
	start = 0

	for w in split_text:
		# print(w, start, start + len(tok_lens[w]))
		if w not in vocab_idx:
			vocab_idx[w] = []

		vocab_idx[w].extend(np.arange(start, start + len(tok_lens[w])))

		start += len(tok_lens[w])

	return vocab_idx

def get_hidden_states(encoded, data_idx, model, layers, static_emb):
	"""Push input IDs through model. Stack and sum `layers` (last four by default).
	Select only those subword token outputs that belong to our word of interest
	and average them."""
	with torch.no_grad():
		output = model(**encoded)

	# Get all hidden states
	states = output.hidden_states
	# Stack and sum all requested layers
	output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

	# Only select the tokens that constitute the requested word

	for w in data_idx:
		static_emb[w] += output[data_idx[w]].sum(dim=0).cpu().numpy()

def chunkify(text, token_lens, length=512):
	chunks = [[]]
	split_text = text.split()
	count = 0
	for word in split_text:
		new_count = count + len(token_lens[word]) + 2 # 2 for [CLS] and [SEP]
		if new_count > length:
			chunks.append([word])
			count = len(token_lens[word])
		else:
			chunks[len(chunks) - 1].append(word)
			count = new_count
	
	return chunks

def constructPrompt(args, init_prompt, main_prompt):
	# GPT APIとローカルAPI（LM Studio/Ollama）は同じメッセージ形式を使用
	if (args.llm == 'gpt' or args.llm == 'local'):
		return [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": main_prompt}]
	else:
		# vLLMなどの場合は文字列形式
		return init_prompt + "\n\n" + main_prompt

def initializeLLM(args):
	args.client = {}

	# vLLMの初期化（必要な場合のみ）
	if LLM is not None and args.llm == 'vllm':
		try:
			args.client['vllm'] = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=4, gpu_memory_utilization=0.95, 
							   max_num_batched_tokens=4096, max_num_seqs=1000, enable_prefix_caching=True)
		except Exception as e:
			print(f"警告: vLLMの初期化に失敗しました: {e}")
			print("GPTを使用します")
			args.llm = 'gpt'

	if args.llm == 'gpt':
		if not openai_key:
			raise ValueError("OPENAI_API_KEY環境変数が設定されていません。.envファイルまたは環境変数で設定してください。")
		args.client[args.llm] = OpenAI(api_key=openai_key)
	elif args.llm == 'local':
		# ローカルAPI（LM Studio/Ollama）の設定
		api_url = args.local_api_url if hasattr(args, 'local_api_url') and args.local_api_url else local_api_url
		api_key = args.local_api_key if hasattr(args, 'local_api_key') and args.local_api_key else local_api_key
		model_name = args.local_model_name if hasattr(args, 'local_model_name') and args.local_model_name else local_model_name
		
		args.client['local'] = OpenAI(
			base_url=api_url,
			api_key=api_key
		)
		args.local_model_name = model_name
		print(f"Local API: {api_url}")
		print(f"Model: {model_name}")
	
	return args

def promptGPT(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	outputs = []
	import time
	import openai
	import json
	
	# 使用するクライアントとモデル名を決定
	if args.llm == 'local':
		client_key = 'local'
		model_name = getattr(args, 'local_model_name', 'qwen/qwen3-vl-30b')
		desc = "Local LLM API"
	else:
		client_key = 'gpt'
		model_name = 'gpt-4o-mini-2024-07-18'
		desc = "GPT API"
	
	for idx, messages in enumerate(tqdm(prompts, desc=desc, ncols=80, ascii=True)):
		max_retries = 3
		retry_delay = 1
		
		for attempt in range(max_retries):
			try:
				# リクエストパラメータの構築
				create_params = {
					'model': model_name,
					'stream': False,
					'messages': messages,
					'temperature': temperature,
					'top_p': top_p,
				}
				
				# max_tokensの設定（0より大きい場合のみ設定）
				if max_new_tokens > 0:
					create_params['max_tokens'] = max_new_tokens
				
				# JSONモードの設定
				# Alibaba Cloud Model StudioのQwen APIリファレンスに基づき、
				# OpenAI互換APIではresponse_formatがサポートされている
				# GPT API使用時: json_object
				# LM Studio経由のQwenモデル: json_schema（スキーマ定義が必要）
				if json_mode:
					if args.llm == 'local':
						# ローカルAPI（LM Studio/Ollama）使用時はjson_schemaを使用
						# schemaが提供されている場合は、JSONスキーマを生成
						if schema is not None:
							try:
								# PydanticスキーマからJSONスキーマを生成
								json_schema = schema.model_json_schema()
								create_params['response_format'] = {
									"type": "json_schema",
									"json_schema": {
										"name": schema.__name__ if hasattr(schema, '__name__') else "response",
										"strict": True,
										"schema": json_schema
									}
								}
							except Exception as e:
								# スキーマ生成に失敗した場合は、プロンプト内でJSON形式を指定
								print(f"警告: JSONスキーマの生成に失敗しました: {e}。プロンプト内でJSON形式を指定します。")
								pass
						else:
							# schemaが提供されていない場合は、プロンプト内でJSON形式を指定
							pass
					else:
						# GPT API使用時はjson_objectを使用
						create_params['response_format'] = {"type": "json_object"}
				
				response = args.client[client_key].chat.completions.create(**create_params)
				outputs.append(response.choices[0].message.content)
				break  # 成功したらループを抜ける
			except openai.RateLimitError as e:
				# ローカルAPIではレート制限エラーは発生しないが、互換性のため残す
				if attempt < max_retries - 1:
					wait_time = retry_delay * (2 ** attempt)  # 指数バックオフ
					print(f"\n警告: レート制限エラー (件数 {idx+1}/{len(prompts)})。{wait_time}秒待機してリトライします...")
					time.sleep(wait_time)
				else:
					print(f"\nエラー: レート制限エラーが{max_retries}回続きました。処理を中断します。")
					raise
			except openai.APIError as e:
				if attempt < max_retries - 1:
					wait_time = retry_delay * (2 ** attempt)
					print(f"\n警告: APIエラー (件数 {idx+1}/{len(prompts)}): {e}")
					if hasattr(e, 'response') and e.response is not None:
						try:
							error_body = e.response.text
							print(f"エラー詳細: {error_body}")
						except:
							pass
					print(f"{wait_time}秒待機してリトライします...")
					time.sleep(wait_time)
				else:
					print(f"\nエラー: APIエラーが{max_retries}回続きました (件数 {idx+1}/{len(prompts)}): {e}")
					if hasattr(e, 'response') and e.response is not None:
						try:
							error_body = e.response.text
							print(f"エラー詳細: {error_body}")
						except:
							pass
					raise
			except Exception as e:
				print(f"\nエラー: 予期しないエラーが発生しました (件数 {idx+1}/{len(prompts)}): {e}")
				raise
	return outputs

def promptLlamaVLLM(args, prompts, schema=None, max_new_tokens=1024, temperature=0.1, top_p=0.99):
    if LLM is None or SamplingParams is None:
        raise ImportError("vLLMがインストールされていません。GPTを使用するか、vLLMをインストールしてください。")
    if schema is None:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    else:
        if GuidedDecodingParams is None:
            raise ImportError("vLLMのGuidedDecodingParamsが利用できません。")
        guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, 
                                    guided_decoding=guided_decoding_params)
    generations = args.client['vllm'].generate(prompts, sampling_params)
    
    outputs = []
    for gen in generations:
        outputs.append(gen.outputs[0].text)

    return outputs

def promptLLM(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	if args.llm == 'gpt' or args.llm == 'local':
		return promptGPT(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
	else:
		return promptLlamaVLLM(args, prompts, schema, max_new_tokens, temperature, top_p)
	