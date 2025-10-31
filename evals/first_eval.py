import os, uuid, time
from dotenv import load_dotenv
from langsmith import evaluate, Client
from src.llms import GoogleGenAILLM
from src.models.agent_model import AgentManager

load_dotenv()

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
SOURCE_DATASET = "icd10 dataset"
SAMPLED_NAME = "icd10 dataset — sample(100)"
SAMPLE_SIZE = 100

# --- Build your agent ---
llm = GoogleGenAILLM(model_name="gemini-2.5-flash", temperature=0.0)
agent = AgentManager(llm=llm)

# --- Callable used by `evaluate` ---
def agent_invoke(example: dict):
    thread_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    res = agent.chat(example["input"], thread_id=thread_id)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_text = (res.get("answer") or "").strip()
    return {"answer": pred_text, "latency_ms": latency_ms}

# --- Evaluators ---
def f1_single_label(outputs, reference_outputs):
    pred = outputs if isinstance(outputs, str) else (outputs.get("answer") or "")
    if isinstance(reference_outputs, str):
        ref = reference_outputs
    elif isinstance(reference_outputs, dict):
        ref = reference_outputs.get("output") or reference_outputs.get("answer") or ""
    else:
        ref = str(reference_outputs)
    score = 1.0 if (ref and pred == ref) else 0.0
    return {"key": "f1", "score": score, "pred": pred, "ref": ref}

def exact_match(outputs, reference_outputs):
    pred = outputs if isinstance(outputs, str) else (outputs.get("answer") or "")
    if isinstance(reference_outputs, str):
        ref = reference_outputs
    elif isinstance(reference_outputs, dict):
        ref = reference_outputs.get("output") or reference_outputs.get("answer") or ""
    else:
        ref = str(reference_outputs)
    return {"key": "exact_match", "score": 1.0 if pred == ref else 0.0}

def latency_metric(outputs, _reference_outputs):
    latency = float("nan")
    if isinstance(outputs, dict):
        latency = outputs.get("latency_ms", float("nan"))
    return {"key": "latency_ms", "score": latency}

# --- Build a sampled dataset with the first N examples ---
src_ds = client.read_dataset(dataset_name=SOURCE_DATASET)
examples_iter = client.list_examples(dataset_id=src_ds.id)

# Create (or reuse) the sampled dataset
try:
    sampled_ds = client.read_dataset(dataset_name=SAMPLED_NAME)
    print(f"Prepared sampled dataset '{SAMPLED_NAME}'.")
except Exception:
    sampled_ds = client.create_dataset(dataset_name=SAMPLED_NAME, description=f"Sample of {SOURCE_DATASET} (first {SAMPLE_SIZE})")

# If you want to avoid duplicates across reruns, you can skip creation if it already has items.
"""
count = 0
for ex in examples_iter:
    if count >= SAMPLE_SIZE:
        break
    inp = ex.inputs.get("input") if isinstance(ex.inputs, dict) else ex.inputs
    ref = ex.outputs.get("output") if isinstance(ex.outputs, dict) else ex.outputs
    client.create_example(
        inputs={"input": inp},
        outputs={"output": (ref or "").strip()},
        dataset_id=sampled_ds.id,
    )
    count += 1

print(f"Prepared sampled dataset '{SAMPLED_NAME}' with {count} examples.")
"""

# --- Run evaluation on the sampled dataset ---
evaluate(
    agent_invoke,
    data=SAMPLED_NAME,  # <-- pass dataset name, not a list
    evaluators=[f1_single_label, exact_match, latency_metric],
    experiment_prefix="icd10-dataset-experiment first 100- with guidelines",
    client=client,
)
