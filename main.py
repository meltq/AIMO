from vllm import LLM, SamplingParams
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = LLM(
    model="/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/1",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.96,
    max_model_len=32000,
)

import time

base = '''Please use chained reasoning to put the answer in \\boxed{}.
Please reflect and verify while reasoning and put the answer in \\boxed{}.
Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.
You are a helpful and reflective maths assistant, please reason step by step to put the answer in \\boxed{}.
You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.
'''
question = r'Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?\n\n'

prompts = [base+question]

a = time.time()

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=32768,
    n=3
)
responses = llm.generate(prompts, sampling_params)

print(time.time()-a)

for response in responses:
    print(response.outputs[0].text)
