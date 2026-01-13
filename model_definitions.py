import os
from tqdm import tqdm
import openai
from openai import OpenAI, AsyncOpenAI
import asyncio
import time

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

# 複数のAPIキーをサポート（複数の方法に対応）
openai_keys = []

# 方法1: OPENAI_API_KEYS（改行区切りまたはカンマ区切り）
openai_keys_str = os.getenv('OPENAI_API_KEYS', '')
if openai_keys_str:
    # 改行区切りを優先、なければカンマ区切り
    if '\n' in openai_keys_str:
        openai_keys = [key.strip() for key in openai_keys_str.split('\n') if key.strip()]
    else:
        openai_keys = [key.strip() for key in openai_keys_str.split(',') if key.strip()]

# 方法2: OPENAI_API_KEY_1, OPENAI_API_KEY_2, ... の形式
if not openai_keys:
    api_key_index = 1
    while True:
        key = os.getenv(f'OPENAI_API_KEY_{api_key_index}')
        if key:
            openai_keys.append(key.strip())
            api_key_index += 1
        else:
            break

# 方法3: 単一のOPENAI_API_KEY
if not openai_keys and openai_key:
    openai_keys = [openai_key]
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
		# 複数のAPIキーをサポート
		if hasattr(args, 'openai_api_keys') and args.openai_api_keys:
			api_keys = args.openai_api_keys
		elif openai_keys:
			api_keys = openai_keys
		elif openai_key:
			api_keys = [openai_key]
		else:
			raise ValueError("OPENAI_API_KEYまたはOPENAI_API_KEYS環境変数が設定されていません。.envファイルまたは環境変数で設定してください。")
		
		# 複数のAPIキーがある場合は、各キーに対してクライアントを作成
		if len(api_keys) > 1:
			args.client[args.llm] = [OpenAI(api_key=key) for key in api_keys]
			args.openai_api_keys = api_keys
			print(f"✓ 並列実行モード: {len(api_keys)}個のAPIキーを使用します")
			for i, key in enumerate(api_keys, 1):
				# APIキーの最初と最後の数文字だけ表示（セキュリティのため）
				masked_key = f"{key[:7]}...{key[-4:]}" if len(key) > 11 else "***"
				print(f"  - APIキー {i}: {masked_key}")
		else:
			args.client[args.llm] = OpenAI(api_key=api_keys[0])
			args.openai_api_keys = api_keys
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

async def _make_api_request(async_client, create_params, idx, total, api_key_index=None):
	"""単一のAPIリクエストを非同期で実行"""
	try:
		response = await async_client.chat.completions.create(**create_params)
		return (idx, response.choices[0].message.content, None, api_key_index, None)
	except Exception as e:
		# レート制限エラーの場合、待機時間を抽出
		wait_time = None
		if isinstance(e, openai.RateLimitError):
			error_message = str(e)
			# "Please try again in X.XXXs" の形式から待機時間を抽出
			import re
			match = re.search(r'try again in ([\d.]+)s', error_message, re.IGNORECASE)
			if match:
				wait_time = float(match.group(1))
		return (idx, None, e, api_key_index, wait_time)

async def promptGPT_parallel_async(args, prompts, api_keys, model_name, schema, max_new_tokens, json_mode, temperature, top_p):
	"""複数のAPIキーを使って並列リクエストを送信（非同期版）"""
	from itertools import cycle
	
	# 非同期クライアントを作成
	async_clients = [AsyncOpenAI(api_key=key) for key in api_keys]
	client_cycle = cycle(range(len(async_clients)))  # インデックスを循環
	
	outputs = [None] * len(prompts)
	
	# リクエストパラメータを構築
	create_params_list = []
	for messages in prompts:
		create_params = {
			'model': model_name,
			'stream': False,
			'messages': messages,
			'temperature': temperature,
			'top_p': top_p,
		}
		
		if max_new_tokens > 0:
			create_params['max_tokens'] = max_new_tokens
		
		if json_mode:
			create_params['response_format'] = {"type": "json_object"}
		
		create_params_list.append(create_params)
	
	print(f"\n{'='*60}")
	print(f"🚀 並列実行開始")
	print(f"   - リクエスト数: {len(prompts)}件")
	print(f"   - APIキー数: {len(api_keys)}個")
	print(f"   - モデル: {model_name}")
	print(f"   - 最大トークン: {max_new_tokens}")
	print(f"{'='*60}")
	
	# プロンプトの内容を要約して表示（最初の3件のみ）
	if len(prompts) > 0:
		print(f"\n📝 処理内容の例（最初の3件）:")
		for i in range(min(3, len(prompts))):
			messages = prompts[i]
			user_content = messages[-1]['content'] if messages and isinstance(messages[-1], dict) else str(messages)[:100]
			preview = user_content[:150] + "..." if len(user_content) > 150 else user_content
			print(f"   [{i+1}] {preview}")
		if len(prompts) > 3:
			print(f"   ... 他 {len(prompts) - 3}件")
	
	# リクエスト間のインターバル（秒）を環境変数から取得、デフォルトは1.5秒
	request_interval = float(os.getenv('OPENAI_REQUEST_INTERVAL', '1.5'))
	
	print(f"\n⏳ 並列リクエスト送信中（同時実行数: {len(api_keys)}件、インターバル: {request_interval}秒）...")
	
	# セマフォを使って同時実行数をAPIキー数に制限
	semaphore = asyncio.Semaphore(len(api_keys))
	start_time = time.time()
	results_dict = {}  # インデックスをキーとして結果を保存
	rate_limit_errors = []  # レート制限エラーを記録
	
	# 各APIキーごとの最後のリクエスト時刻を記録
	last_request_time = {i: 0.0 for i in range(len(api_keys))}
	# 各APIキーごとのロック（同じAPIキーへの同時リクエストを防ぐ）
	api_key_locks = {i: asyncio.Lock() for i in range(len(api_keys))}
	
	async def process_with_semaphore(idx, create_params, api_key_idx):
		"""セマフォを使って同時実行数を制限しながらリクエストを実行"""
		async with semaphore:
			# 同じAPIキーへの同時リクエストを防ぐ
			async with api_key_locks[api_key_idx]:
				# 最後のリクエストからの経過時間を確認
				elapsed_since_last = time.time() - last_request_time[api_key_idx]
				if elapsed_since_last < request_interval:
					# インターバル時間が経過していない場合は待機
					wait_time = request_interval - elapsed_since_last
					await asyncio.sleep(wait_time)
				
				# リクエストを実行
				async_client = async_clients[api_key_idx]
				result = await _make_api_request(async_client, create_params, idx, len(prompts), api_key_idx + 1)
				
				# 最後のリクエスト時刻を更新
				last_request_time[api_key_idx] = time.time()
				
				return result
	
	# すべてのリクエストをタスクとして作成（セマフォで制限される）
	tasks = []
	for idx, create_params in enumerate(create_params_list):
		api_key_idx = next(client_cycle)  # ラウンドロビンでAPIキーを選択
		tasks.append(process_with_semaphore(idx, create_params, api_key_idx))
	
	# タスクを順序を保証して処理（gatherを使用して順序を保証）
	# 進捗表示のため、成功したタスクのみを追跡
	success_tasks = set()
	
	# タスクを実行
	results_list = await asyncio.gather(*tasks, return_exceptions=True)
	
	# 結果をインデックス順に処理（進捗表示付き）
	for idx, result in enumerate(results_list):
		if isinstance(result, Exception):
			raise result
		idx_result, content, error, api_key_idx, wait_time = result
		results_dict[idx_result] = (idx_result, content, error, api_key_idx, wait_time)
		
		# 成功したタスクのみをカウント
		if error is None:
			success_tasks.add(idx_result)
			success_count = len(success_tasks)
			elapsed = time.time() - start_time
			rate = success_count / elapsed if elapsed > 0 else 0
			remaining = len(prompts) - success_count
			eta = remaining / rate if rate > 0 else 0
			print(f"  ✓ [{success_count:4d}/{len(prompts)}] 成功 ({elapsed:.1f}秒経過, 残り約{eta:.1f}秒)", end='\r')
		elif isinstance(error, openai.RateLimitError):
			# レート制限エラーを記録
			rate_limit_errors.append((idx_result, api_key_idx, wait_time, error))
	
	print()  # 改行
	
	# レート制限エラーが発生した場合の処理（複数回リトライ）
	max_retries = int(os.getenv('OPENAI_MAX_RETRIES', '3'))  # 最大リトライ回数（デフォルト3回）
	retry_attempt = 0
	failed_requests = rate_limit_errors.copy()  # 失敗したリクエストを追跡
	
	while failed_requests and retry_attempt < max_retries:
		retry_attempt += 1
		print(f"\n⚠️  レート制限エラーが{len(failed_requests)}件発生しました（リトライ {retry_attempt}/{max_retries}）")
		
		# 最大の待機時間を取得
		max_wait_time = max([w for _, _, w, _ in failed_requests if w is not None], default=5.0)
		if max_wait_time:
			print(f"   ⏳ {max_wait_time:.1f}秒待機してからリトライします...")
			await asyncio.sleep(max_wait_time + 1)  # 少し余裕を持たせる
		
		# 失敗したリクエストをリトライ
		print(f"   🔄 {len(failed_requests)}件のリクエストをリトライします...")
		retry_tasks = []
		for idx, api_key_idx, _, error in failed_requests:
			# 別のAPIキーでリトライ（ラウンドロビン）
			retry_key_idx = next(client_cycle)
			retry_params = create_params_list[idx]
			# セマフォを使って同時実行数を制限
			retry_tasks.append(process_with_semaphore(idx, retry_params, retry_key_idx))
		
		# リトライを並列実行（セマフォで制限される、順序を保証）
		retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)
		
		# リトライ結果を処理
		failed_requests = []  # 次のリトライ用にリセット
		retry_success_count = 0
		for retry_result in retry_results:
			if isinstance(retry_result, Exception):
				raise retry_result
			idx, content, error, api_key_idx, wait_time = retry_result
			if error is None:
				results_dict[idx] = (idx, content, None, api_key_idx, wait_time)
				retry_success_count += 1
				print(f"  ✓ リトライ成功 [{retry_success_count}/{len(retry_tasks)}] (件数 {idx+1}, APIキー{api_key_idx})", end='\r')
			else:
				# リトライも失敗した場合は次のリトライに追加
				if isinstance(error, openai.RateLimitError):
					failed_requests.append((idx, api_key_idx, wait_time, error))
				else:
					# レート制限以外のエラーは即座に例外を発生
					print(f"\n❌ リトライ失敗: APIエラー (件数 {idx+1}, APIキー {api_key_idx}): {error}")
					raise error
		print()  # 改行
		
		# すべて成功した場合はループを抜ける
		if not failed_requests:
			break
	
	# 最大リトライ回数に達しても失敗した場合は空のJSONを返す
	if failed_requests:
		print(f"\n⚠️  最大リトライ回数（{max_retries}回）に達しました。{len(failed_requests)}件のリクエストは失敗しました。")
		for idx, api_key_idx, _, error in failed_requests:
			results_dict[idx] = (idx, "{}", error, api_key_idx, None)
	
	# 結果をインデックス順に並べ替え
	results = [results_dict[i] for i in range(len(prompts))]
	
	# 結果を処理
	success_count = 0
	error_count = 0
	api_key_usage = {i+1: 0 for i in range(len(api_keys))}  # 各APIキーの使用回数を記録
	
	for result in results:
		if result is None:
			continue
		idx, content, error, api_key_idx, wait_time = result
		if error is None:
			outputs[idx] = content
			success_count += 1
			api_key_usage[api_key_idx] = api_key_usage.get(api_key_idx, 0) + 1
		else:
			error_count += 1
			# エラーの詳細を表示
			if isinstance(error, openai.RateLimitError):
				error_msg = str(error)
				# エラーメッセージから詳細を抽出
				if 'tokens per min' in error_msg.lower():
					print(f"\n❌ レート制限エラー (件数 {idx+1}): TPM制限に達しました")
				elif 'requests per min' in error_msg.lower():
					print(f"\n❌ レート制限エラー (件数 {idx+1}): RPM制限に達しました")
				else:
					print(f"\n❌ レート制限エラー (件数 {idx+1}): {error_msg[:200]}")
				# レート制限エラーの場合、空のJSONオブジェクトを返して処理を続行
				outputs[idx] = "{}"
			else:
				print(f"\n❌ APIエラー (件数 {idx+1}, APIキー {api_key_idx}): {error}")
				# レート制限以外のエラーは即座に例外を発生
				print(f"\n❌ APIエラー (件数 {idx+1}, APIキー {api_key_idx}): {error}")
				raise error
	
	# 最終結果を表示
	total_time = time.time() - start_time
	print(f"\n{'='*60}")
	print(f"✅ 並列実行完了")
	print(f"   - 成功: {success_count}/{len(prompts)}件")
	if error_count > 0:
		print(f"   - エラー: {error_count}件")
	print(f"   - 処理時間: {total_time:.2f}秒")
	print(f"   - 平均速度: {success_count/total_time:.2f}件/秒" if total_time > 0 else "   - 平均速度: N/A")
	print(f"\n📊 APIキー使用状況:")
	for key_idx, count in sorted(api_key_usage.items()):
		percentage = (count / success_count * 100) if success_count > 0 else 0
		print(f"   - APIキー {key_idx}: {count}件 ({percentage:.1f}%)")
	print(f"{'='*60}\n")
	
	# クライアントをクローズ
	for client in async_clients:
		await client.close()
	
	return outputs

def promptGPT_parallel(args, prompts, clients, api_keys, model_name, schema, max_new_tokens, json_mode, temperature, top_p):
	"""複数のAPIキーを使って並列リクエストを送信（同期ラッパー）"""
	outputs = asyncio.run(promptGPT_parallel_async(
		args, prompts, api_keys, model_name, schema, max_new_tokens, json_mode, temperature, top_p
	))
	return outputs

def promptGPT(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	import time
	import openai
	import json
	import asyncio
	from itertools import cycle
	
	# 使用するクライアントとモデル名を決定
	if args.llm == 'local':
		client_key = 'local'
		model_name = getattr(args, 'local_model_name', 'qwen/qwen3-vl-30b')
		desc = "Local LLM API"
		use_parallel = False
		client = args.client[client_key]
	else:
		client_key = 'gpt'
		model_name = 'gpt-4o-mini-2024-07-18'
		desc = "GPT API"
		# 複数のAPIキーがある場合でも、プロンプトが1件の場合は順次処理を使用（STEP2など）
		# プロンプトが複数ある場合のみ並列処理を使用
		clients = args.client[client_key]
		if isinstance(clients, list) and len(clients) > 1 and len(prompts) > 1:
			use_parallel = True
			api_keys = args.openai_api_keys
			client = None  # 並列処理では使用しない
		else:
			use_parallel = False
			if isinstance(clients, list):
				client = clients[0]
			else:
				client = clients
	
	# 並列処理を使用する場合
	if use_parallel:
		return promptGPT_parallel(args, prompts, clients, api_keys, model_name, schema, max_new_tokens, json_mode, temperature, top_p)
	
	# 順次処理（既存の実装）
	outputs = []
	# 複数のAPIキーがある場合は、順次処理でも別のキーでリトライできるようにする
	available_clients = []
	use_multiple_keys = False
	if args.llm == 'gpt' and hasattr(args, 'openai_api_keys') and args.openai_api_keys and len(args.openai_api_keys) > 1:
		# 複数のAPIキーがある場合
		available_clients = [OpenAI(api_key=key) for key in args.openai_api_keys]
		client_cycle = cycle(range(len(available_clients)))  # インデックスを循環
		use_multiple_keys = True
		print(f"✓ 順次処理モード（複数APIキー対応）: {len(available_clients)}個のAPIキーを使用します")
	else:
		# 単一のAPIキーの場合
		available_clients = [client]
		client_cycle = cycle([0])  # インデックス0のみ
	
	for idx, messages in enumerate(tqdm(prompts, desc=desc, ncols=80, ascii=True)):
		max_retries = len(available_clients) * 3  # 各クライアントで3回までリトライ
		retry_delay = 1
		current_key_idx = next(client_cycle)
		current_client = available_clients[current_key_idx]
		
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
				
				response = current_client.chat.completions.create(**create_params)
				outputs.append(response.choices[0].message.content)
				break  # 成功したらループを抜ける
			except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
				# APIキーが無効な場合、別のキーでリトライ
				if use_multiple_keys:
					print(f"\n⚠️  APIキーが無効です (件数 {idx+1}/{len(prompts)}, APIキー {current_key_idx + 1})。別のAPIキーでリトライします...")
					current_key_idx = next(client_cycle)
					current_client = available_clients[current_key_idx]
					if attempt < max_retries - 1:
						continue
				print(f"\nエラー: APIキーが無効です (件数 {idx+1}/{len(prompts)}): {e}")
				if hasattr(e, 'response') and e.response is not None:
					try:
						error_body = e.response.text
						print(f"エラー詳細: {error_body}")
					except:
						pass
				raise
			except openai.RateLimitError as e:
				# ローカルAPIではレート制限エラーは発生しないが、互換性のため残す
				# 複数のAPIキーがある場合は別のキーでリトライ
				if use_multiple_keys:
					print(f"\n⚠️  レート制限エラー (件数 {idx+1}/{len(prompts)}, APIキー {current_key_idx + 1})。別のAPIキーでリトライします...")
					current_key_idx = next(client_cycle)
					current_client = available_clients[current_key_idx]
					if attempt < max_retries - 1:
						continue
				if attempt < max_retries - 1:
					wait_time = retry_delay * (2 ** attempt)  # 指数バックオフ
					print(f"\n警告: レート制限エラー (件数 {idx+1}/{len(prompts)})。{wait_time}秒待機してリトライします...")
					time.sleep(wait_time)
				else:
					print(f"\nエラー: レート制限エラーが{max_retries}回続きました。処理を中断します。")
					raise
			except openai.APIError as e:
				# 複数のAPIキーがある場合は別のキーでリトライ
				if use_multiple_keys and attempt < max_retries - 1:
					print(f"\n⚠️  APIエラー (件数 {idx+1}/{len(prompts)}, APIキー {current_key_idx + 1}): {e}。別のAPIキーでリトライします...")
					current_key_idx = next(client_cycle)
					current_client = available_clients[current_key_idx]
					continue
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
	