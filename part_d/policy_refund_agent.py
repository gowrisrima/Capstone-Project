"""Part D: offline policy RAG and resumable, guarded refund agent.
Run from the Capstone_Project root: python part_d/policy_refund_agent.py
"""
import csv
import json
import re
import sqlite3
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mock_llm

DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
STATE = ROOT / 'state'
OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)
FALLBACK_TEXT = 'I could not find that information in the policy documents.'


def make_chunks(overlap=10, size=40):
    chunks = []
    for path in sorted((DATA / 'policies').glob('*.txt')):
        words = path.read_text(encoding='utf-8').split()
        for start in range(0, len(words), size - overlap):
            chunks.append({'source': path.name, 'text': ' '.join(words[start:start + size])})
            if start + size >= len(words):
                break
    return chunks


class PolicyRetriever:
    def __init__(self, overlap=10):
        self.chunks = make_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c['text'] for c in self.chunks])

    def retrieve(self, query, k=3):
        similarities = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = np.argsort(-similarities, kind='stable')[:k]
        return [{'source': self.chunks[i]['source'], 'score': float(similarities[i]), 'text': self.chunks[i]['text']} for i in order]

    def answer(self, question):
        hits = self.retrieve(question, 3)
        context = '\n'.join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [
            {'role': 'system', 'content': 'POLICY ASSISTANT. Answer only using supplied context. If the answer is not in context, say: I could not find that information in the policy documents.'},
            {'role': 'user', 'content': f'CONTEXT:\n{context}\nQUESTION:\n{question}'},
        ]
        response = mock_llm.chat(messages)
        answer_text = response.choices[0].message.content
        return answer_text, [h['source'] for h in hits]


RETRIEVER = None


def retrieve(query, k=3):
    global RETRIEVER
    if RETRIEVER is None:
        RETRIEVER = PolicyRetriever()
    return RETRIEVER.retrieve(query, k)


def answer(question):
    global RETRIEVER
    if RETRIEVER is None:
        RETRIEVER = PolicyRetriever()
    return RETRIEVER.answer(question)


def evaluate(retriever):
    rows = list(csv.DictReader((DATA / 'raw' / 'policy_eval.csv').open(encoding='utf-8-sig', newline='')))
    source_hits = fact_hits = fallback_hits = 0
    failures = []
    for row in rows:
        reply, sources = retriever.answer(row['question'])
        if row['expected_doc'].strip().upper() != 'NONE':
            source_hits += row['expected_doc'] in sources
            fact_hits += row['expected_fact'] in reply
            if row['expected_fact'] not in reply:
                failures.append((row['question'], row['expected_fact'], reply))
        else:
            fallback_hits += reply.startswith('I could not find')
    return source_hits, fact_hits, fallback_hits, failures


def get_order(order_id):
    with sqlite3.connect(DATA / 'instamart.db') as con:
        row = con.execute('SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?', (order_id,)).fetchone()
    return dict(zip(('order_id', 'status', 'order_amount', 'delivery_minutes'), row)) if row else {'error': 'order not found'}


def search_policy(query):
    return {'results': retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount or 0)
    duration = float(delivery_minutes or 0)
    return {'refund_inr': round(amount, 2) if duration > 45 else 0.0}


def _ledger(tool_name, order_id, refund_inr):
    record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'tool': tool_name, 'order_id': order_id, 'refund_inr': float(refund_inr)}
    with (OUT / 'refund_ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    return {'refund_inr': float(refund_inr)}


def issue_refund(order_id, refund_inr):
    return _ledger('issue_refund', order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger('escalate_to_human', order_id, refund_inr)


TOOLS_SPEC = [
    {'name': 'get_order', 'description': 'Look up an order in the cleaned SQLite database.', 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}}, 'required': ['order_id']}},
    {'name': 'search_policy', 'description': 'Retrieve policy passages related to a question.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
    {'name': 'calculate_refund', 'description': 'Calculate the late-delivery refund amount.', 'parameters': {'type': 'object', 'properties': {'order_amount': {'type': 'number'}, 'delivery_minutes': {'type': 'number'}}, 'required': ['order_amount', 'delivery_minutes']}},
    {'name': 'issue_refund', 'description': 'Record an approved automatic refund in the ledger.', 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}, 'refund_inr': {'type': 'number'}}, 'required': ['order_id', 'refund_inr']}},
    {'name': 'escalate_to_human', 'description': 'Record a high-value refund for human approval.', 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}, 'refund_inr': {'type': 'number'}}, 'required': ['order_id', 'refund_inr']}},
]
TOOL_MAP = {'get_order': get_order, 'search_policy': search_policy, 'calculate_refund': calculate_refund, 'issue_refund': issue_refund, 'escalate_to_human': escalate_to_human}


def _checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, steps):
    state = {'run_id': run_id, 'scenario_id': scenario_id, 'messages': messages, 'tokens_used': tokens_used, 'llm_calls': llm_calls, 'step_number': len(steps), 'tools_executed': [s['tool'] for s in steps], 'steps': steps}
    (STATE / f'{run_id}.json').write_text(json.dumps(state, indent=2), encoding='utf-8')


def _result(status, final_answer, steps, llm_calls, tokens_used, scenario_id):
    return {'scenario_id': scenario_id, 'status': status, 'final_answer': final_answer, 'steps': steps, 'llm_calls': llm_calls, 'tokens_used': tokens_used}


def _loop(messages, scenario_id, run_id, steps, llm_calls, tokens_used, max_iterations, token_limit, time_limit_seconds, crash_after, started):
    while llm_calls < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            return _result('FALLBACK_TIMEOUT', 'Human handoff: agent exceeded the time limit.', steps, llm_calls, tokens_used, scenario_id)
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return _result('FALLBACK_TOKEN_LIMIT', 'Human handoff: agent exceeded the token budget.', steps, llm_calls, tokens_used, scenario_id)
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            final = message.content or ''
            return _result(final.split(':', 1)[0].strip(), final, steps, llm_calls, tokens_used, scenario_id)
        serialized = []
        for call in calls:
            args = call.arguments
            if isinstance(args, str):
                args = json.loads(args)
            serialized.append({'id': call.id, 'name': call.name, 'arguments': args})
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': serialized})
        for call in serialized:
            name, args = call['name'], call['arguments']
            if name not in TOOL_MAP:
                observation = {'error': 'tool not allowed'}
            else:
                # Enforce the action safety rules independently of the LLM.
                if name in ('issue_refund', 'escalate_to_human'):
                    order = get_order(args.get('order_id', ''))
                    requested = float(args.get('refund_inr', 0))
                    valid_amount = ('error' not in order and order['status'] == 'completed' and
                                    abs(requested - calculate_refund(order['order_amount'], order['delivery_minutes'])['refund_inr']) < .01 and requested > 0)
                    valid_route = (name == 'issue_refund' and requested <= 1000) or (name == 'escalate_to_human' and requested > 1000)
                    observation = TOOL_MAP[name](**args) if valid_amount and valid_route else {'error': 'refund action not authorized'}
                else:
                    observation = TOOL_MAP[name](**args)
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name, 'content': json.dumps(observation)})
            steps.append({'tool': name, 'arguments': args, 'observation': observation})
            _checkpoint(run_id, scenario_id, messages, tokens_used, llm_calls, steps)
            if crash_after is not None and len(steps) == crash_after:
                raise RuntimeError('Simulated crash after saved checkpoint')
    return _result('FALLBACK_MAX_ITERATIONS', 'Human handoff: agent reached the maximum model-call limit.', steps, llm_calls, tokens_used, scenario_id)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after=None):
    run_id = run_id or f'{scenario_id}_{uuid.uuid4().hex[:10]}'
    messages = [{'role': 'system', 'content': 'AGENT. Use tools to look up orders, search refund policy, calculate refund, and only issue eligible refunds or escalate high-value refunds. Follow all tool observations.'}, {'role': 'user', 'content': user_message}]
    return _loop(messages, scenario_id, run_id, [], 0, 0, max_iterations, token_limit, time_limit_seconds, crash_after, time.monotonic())


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    state = json.loads((STATE / f'{run_id}.json').read_text(encoding='utf-8'))
    return _loop(state['messages'], state['scenario_id'], run_id, state['steps'], state['llm_calls'], state['tokens_used'], max_iterations, token_limit, time_limit_seconds, None, time.monotonic())


def main():
    chunks = make_chunks()
    print('CHUNKS:', len(chunks), dict(sorted(Counter(c['source'] for c in chunks).items())))
    print('TOP 3:', [x['source'] for x in retrieve('refund policy late delivery')])
    for overlap in (10, 0):
        sources, facts, fallback, failures = evaluate(PolicyRetriever(overlap))
        print(f'POLICY EVALUATION overlap={overlap}: sources={sources}/12 facts={facts}/12 fallback={fallback}/2')
        if failures:
            print('Example broken question:', failures[0][0], '| expected:', failures[0][1])
    scenarios = list(csv.DictReader((DATA / 'raw' / 'agent_scenarios.csv').open(encoding='utf-8-sig', newline='')))
    (OUT / 'agent_runs.jsonl').write_text('', encoding='utf-8')
    (OUT / 'refund_ledger.jsonl').write_text('', encoding='utf-8')
    print('\nSCENARIOS')
    for row in scenarios:
        result = run_agent(row['message'], row['scenario_id'])
        with (OUT / 'agent_runs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')
        refund = next((s['observation']['refund_inr'] for s in reversed(result['steps']) if s['tool'] in ('issue_refund', 'escalate_to_human') and 'refund_inr' in s['observation']), 0)
        print(row['scenario_id'], result['status'], 'refund=', refund, 'calls=', result['llm_calls'], 'tokens=', result['tokens_used'])
    s1 = scenarios[0]
    print('\nTOKEN DEMO:', run_agent(s1['message'], 'S1_TOKEN_DEMO', token_limit=200)['status'])
    print('TIMEOUT DEMO:', run_agent(s1['message'], 'S1_TIMEOUT_DEMO', time_limit_seconds=0)['status'])
    run_id = 'S1_crash_resume_demo'
    try:
        run_agent(s1['message'], 'S1_RESUME', run_id=run_id, crash_after=2)
    except RuntimeError as e:
        print('CRASH DEMO:', e)
    resumed = resume_agent(run_id)
    print('RESUMED:', resumed['status'], 'tools=', [s['tool'] for s in resumed['steps']])
    print('OUTPUTS:', OUT / 'agent_runs.jsonl', OUT / 'refund_ledger.jsonl', STATE)


if __name__ == '__main__':
    main()
