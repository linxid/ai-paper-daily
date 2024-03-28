import PyPDF2
from PyPDF2 import PdfReader
import pdb
from zhipuai import ZhipuAI

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    
    # 遍历每一页
    for page in reader.pages:
        text += page.extract_text() + "\n"  # 提取当前页的文本

    return text

def summary_abstract_with_gpt(abstract, model="gpt-3"):
    prompt = f"""
    I have an abstract from a research paper that I need summarized to grasp the main findings and contributions more efficiently. Could you provide a concise summary highlighting the key points, findings, and implications? Here's the abstract:
    {abstract}
    Please summarize the essential aspects of this abstract in a more accessible and condensed form.
    """
    client = ZhipuAI(api_key="c9ce61c6e40ea05351e2365cee3698ec.7s8VTjRYVzO8dReS") # 填写您自己的APIKey
    if model == "ZHIPU":
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        pdb.set_trace()
        return response.choices[0].message.content

if __name__ == '__main__':
    file_path = './assets/2403.09611.pdf'
    text = load_pdf(file_path)
    abstract = """
    Recent advancements in Multimodal Large Language Models (MLLMs) have
    been utilizing Visual Prompt Generators (VPGs) to convert visual features into
    tokens that LLMs can recognize. This is achieved by training the VPGs on
    millions of image-caption pairs, where the VPG-generated tokens of images are
    fed into a frozen LLM to generate the corresponding captions. However, this
    image-captioning based training objective inherently biases the VPG to concentrate solely on the primary visual contents sufficient for caption generation, often
    neglecting other visual details. This shortcoming results in MLLMs’ underperformance in comprehending demonstrative instructions consisting of multiple, interleaved, and multimodal instructions that demonstrate the required context to
    complete a task. To address this issue, we introduce a generic and lightweight Visual Prompt Generator Complete module (VPG-C), which can infer and complete
    the missing details essential for comprehending demonstrative instructions. Further, we propose a synthetic discriminative training strategy to fine-tune VPG-C,
    eliminating the need for supervised demonstrative instructions. As for evaluation,
    we build DEMON, a comprehensive benchmark for demonstrative instruction understanding. Synthetically trained with the proposed strategy, VPG-C achieves
    significantly stronger zero-shot performance across all tasks of DEMON. Further
    evaluation on the MME and OwlEval benchmarks also demonstrate the superiority of VPG-C. Our benchmark, code, and pre-trained models are available at
    https://github.com/DCDmllm/Cheetah.
    """
    # pdb.set_trace()
    summary = summary_abstract_with_gpt(abstract, model="ZHIPU")
    print(summary)
