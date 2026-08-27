"""
召回消融实验: Vector vs BM25 vs Hybrid vs Hybrid+Rerank
- 测试不同切块长度 (80, 200)
- 改进版召回率计算（预处理文本）
- DPO风格对比评估（Pairwise + 交换位置消除偏见）
- 结果输出到 rag_badcase_log.txt
"""
import time
import os
import re
from datetime import datetime
from core.llm import LLM
from core.embedding import Embedding
from core.vector_store import VectorStore
from core.bm25_store import BM25Store
from core.rerank import Rerank
from pypdf import PdfReader


TEST_CASES = [
    {"id": 1, "question": "What is the title of the paper that introduces GPT-3 as a few-shot learner?", "expected_keywords": ["Language Models", "Few-Shot Learners", "GPT-3"], "category": "GPT3"},
    {"id": 2, "question": "How many parameters does GPT-3 have according to the paper?", "expected_keywords": ["175 billion", "parameters", "GPT-3"], "category": "GPT3"},
    {"id": 3, "question": "What type of language model is GPT-3 described as?", "expected_keywords": ["autoregressive", "language model", "GPT-3"], "category": "GPT3"},
    {"id": 4, "question": "Does GPT-3 use gradient updates or fine-tuning during few-shot evaluation?", "expected_keywords": ["without gradient updates", "no fine-tuning", "text interaction"], "category": "GPT3"},
    {"id": 5, "question": "What are the three evaluation settings used for GPT-3 in the paper?", "expected_keywords": ["zero-shot", "one-shot", "few-shot"], "category": "GPT3"},
    {"id": 6, "question": "What does in-context learning mean in the GPT-3 paper?", "expected_keywords": ["context", "demonstrations", "forward pass"], "category": "GPT3"},
    {"id": 7, "question": "Why do the authors compare GPT-3 few-shot learning with human learning?", "expected_keywords": ["few examples", "simple instructions", "humans"], "category": "GPT3"},
    {"id": 8, "question": "What limitation of task-specific fine-tuning motivates the GPT-3 paper?", "expected_keywords": ["task-specific datasets", "thousands", "fine-tuning"], "category": "GPT3"},
    {"id": 9, "question": "What types of NLP tasks does GPT-3 achieve strong performance on?", "expected_keywords": ["translation", "question-answering", "cloze"], "category": "GPT3"},
    {"id": 10, "question": "Which tasks are mentioned as requiring on-the-fly reasoning or domain adaptation?", "expected_keywords": ["unscrambling words", "novel word", "3-digit arithmetic"], "category": "GPT3"},
    {"id": 11, "question": "What does the paper say about larger models and in-context information?", "expected_keywords": ["larger models", "in-context information", "efficient use"], "category": "GPT3"},
    {"id": 12, "question": "What is the purpose of Figure 1.1 in the GPT-3 paper?", "expected_keywords": ["language model meta-learning", "pre-training", "in-context learning"], "category": "GPT3"},
    {"id": 13, "question": "What simple task is used in Figure 1.2 to illustrate in-context learning?", "expected_keywords": ["remove random symbols", "word", "task description"], "category": "GPT3"},
    {"id": 14, "question": "How does the paper define zero-shot learning for GPT-3?", "expected_keywords": ["no demonstrations", "natural language instruction", "zero-shot"], "category": "GPT3"},
    {"id": 15, "question": "How does the paper define one-shot learning for GPT-3?", "expected_keywords": ["one demonstration", "one-shot", "inference time"], "category": "GPT3"},
    {"id": 16, "question": "How does the paper define few-shot learning for GPT-3?", "expected_keywords": ["many demonstrations", "context window", "few-shot"], "category": "GPT3"},
    {"id": 17, "question": "What performance does GPT-3 achieve on CoQA in the few-shot setting?", "expected_keywords": ["CoQA", "85.0 F1", "few-shot"], "category": "GPT3"},
    {"id": 18, "question": "What performance does GPT-3 achieve on TriviaQA in the few-shot setting?", "expected_keywords": ["TriviaQA", "71.2%", "few-shot"], "category": "GPT3"},
    {"id": 19, "question": "What broader impact related to news articles is discussed in the GPT-3 paper?", "expected_keywords": ["synthetic news articles", "human evaluators", "distinguishing"], "category": "GPT3"},
    {"id": 20, "question": "Which types of tasks does GPT-3 still struggle with according to the paper?", "expected_keywords": ["ANLI", "RACE", "QuAC", "limitations"], "category": "GPT3"},
    {"id": 21, "question": "What is the main research topic of the SEM knowledge gain paper?", "expected_keywords": ["search engine marketing", "knowledge gain", "information search behavior model"], "category": "SEM"},
    {"id": 22, "question": "Who is the author of the paper on search engine marketing and knowledge gain?", "expected_keywords": ["Sebastian Schulthei", "Hamburg University", "Applied Sciences"], "category": "SEM"},
    {"id": 23, "question": "What two components does search engine marketing cover in the paper?", "expected_keywords": ["search engine optimization", "paid search marketing", "SEO", "PSM"], "category": "SEM"},
    {"id": 24, "question": "How does the paper define search engine optimization?", "expected_keywords": ["optimizing web pages", "organic search results", "ranking"], "category": "SEM"},
    {"id": 25, "question": "How does the paper define paid search marketing?", "expected_keywords": ["sponsored", "paid results", "search engines"], "category": "SEM"},
    {"id": 26, "question": "Why are SEM measures relevant to user knowledge gain?", "expected_keywords": ["commercial", "political", "motives", "information quality"], "category": "SEM"},
    {"id": 27, "question": "What stakeholder groups influence search engine result pages according to the paper?", "expected_keywords": ["users", "content producers", "search engine marketing"], "category": "SEM"},
    {"id": 28, "question": "What percentage of German Internet users know that Google results can be influenced beyond ads?", "expected_keywords": ["43%", "SEO", "German Internet users"], "category": "SEM"},
    {"id": 29, "question": "What percentage of German Internet users are aware of Google's advertising business model?", "expected_keywords": ["68%", "advertising business model", "Google"], "category": "SEM"},
    {"id": 30, "question": "How is knowledge gain defined in the SEM paper?", "expected_keywords": ["difference", "knowledge before", "knowledge after", "search task"], "category": "SEM"},
    {"id": 31, "question": "What are the two main research questions motivating the SEM doctoral project?", "expected_keywords": ["information quality", "SEM", "documents", "knowledge gain"], "category": "SEM"},
    {"id": 32, "question": "What does RQ1 ask in the SEM paper?", "expected_keywords": ["search engine literacy", "selection behavior", "user"], "category": "SEM"},
    {"id": 33, "question": "What is hypothesis H1a in the SEM paper?", "expected_keywords": ["higher PSM awareness", "fewer ads", "selected"], "category": "SEM"},
    {"id": 34, "question": "What is hypothesis H1b in the SEM paper?", "expected_keywords": ["higher SEO awareness", "wider range", "organic results"], "category": "SEM"},
    {"id": 35, "question": "What intrinsic information quality criteria are mentioned in the paper?", "expected_keywords": ["accuracy", "objectivity", "believability", "reputation"], "category": "SEM"},
    {"id": 36, "question": "What representational information quality criteria are mentioned in the paper?", "expected_keywords": ["interpretability", "ease of understanding", "concise representation", "consistent representation"], "category": "SEM"},
    {"id": 37, "question": "What methods are planned to answer the SEM research questions?", "expected_keywords": ["online experiment", "expert evaluation", "causal relationships"], "category": "SEM"},
    {"id": 38, "question": "What fields are selected for topic collection in the SEM experiment?", "expected_keywords": ["health", "politics", "environment", "consumer protection"], "category": "SEM"},
    {"id": 39, "question": "What are the three main stages of the SEM online experiment procedure?", "expected_keywords": ["pre-experiment survey", "experiment", "post-experiment survey"], "category": "SEM"},
    {"id": 40, "question": "Why is the search engine literacy survey conducted at the end of the SEM experiment?", "expected_keywords": ["not influence", "user behavior", "post-experiment"], "category": "SEM"},
    {"id": 41, "question": "What phenomenon is abbreviated as MIPS in the chemotaxis paper?", "expected_keywords": ["motility-induced phase separation", "MIPS", "dense and dilute phases"], "category": "MIPS"},
    {"id": 42, "question": "Who are the authors of the chemotactic MIPS paper?", "expected_keywords": ["Hongbo Zhao", "Andrej Kosmrlj", "Sujit S. Datta"], "category": "MIPS"},
    {"id": 43, "question": "What institution are the authors of the chemotactic MIPS paper affiliated with?", "expected_keywords": ["Princeton University", "Chemical and Biological Engineering", "Mechanical and Aerospace Engineering"], "category": "MIPS"},
    {"id": 44, "question": "What date is listed on the chemotactic MIPS paper?", "expected_keywords": ["January 31", "2023", "Dated"], "category": "MIPS"},
    {"id": 45, "question": "What does the paper say collective chemotaxis can do to MIPS?", "expected_keywords": ["competes", "arresting", "suppressing", "dynamic instabilities"], "category": "MIPS"},
    {"id": 46, "question": "What is collective chemotaxis in the context of active matter?", "expected_keywords": ["directed motion", "chemical gradient", "particles generate themselves"], "category": "MIPS"},
    {"id": 47, "question": "What kinds of active matter systems are mentioned as examples in the MIPS paper?", "expected_keywords": ["enzymes", "motile microorganisms", "mammalian cells", "robots"], "category": "MIPS"},
    {"id": 48, "question": "What are Active Brownian Particles in the MIPS paper?", "expected_keywords": ["self-propelled", "velocity", "random thermal fluctuations", "reoriented"], "category": "MIPS"},
    {"id": 49, "question": "What is the persistence length of an ABP trajectory proportional to?", "expected_keywords": ["U0", "tau_R", "persistence length"], "category": "MIPS"},
    {"id": 50, "question": "How is the reorientation Peclet number PeR defined?", "expected_keywords": ["PeR", "a", "U0", "tau_R"], "category": "MIPS"},
    {"id": 51, "question": "Why is MIPS surprising compared with passive equilibrium phase separation?", "expected_keywords": ["without attractive interactions", "dense and dilute phases", "out-of-equilibrium"], "category": "MIPS"},
    {"id": 52, "question": "Which classical theory inspires some descriptions of MIPS dynamics?", "expected_keywords": ["Cahn-Hilliard", "phase separation", "passive systems"], "category": "MIPS"},
    {"id": 53, "question": "What are the two contributions to the particle flux J in the governing equation?", "expected_keywords": ["MIPS", "chemotaxis", "flux"], "category": "MIPS"},
    {"id": 54, "question": "What does the chemotactic coefficient chi0 describe?", "expected_keywords": ["ability", "move up", "chemoattractant gradient"], "category": "MIPS"},
    {"id": 55, "question": "What does the chemotactic Peclet number PeC describe?", "expected_keywords": ["competition", "directed chemotaxis", "undirected active diffusion"], "category": "MIPS"},
    {"id": 56, "question": "What is the role of the chemoattractant concentration c in the model?", "expected_keywords": ["diffusible chemical signal", "particles sense", "direct motion"], "category": "MIPS"},
    {"id": 57, "question": "What happens to the homogeneous state when chemotaxis is absent and the system is below the spinodal?", "expected_keywords": ["spontaneously separates", "dense and dilute phases", "spinodal decomposition"], "category": "MIPS"},
    {"id": 58, "question": "What kinds of dynamic behavior can chemotaxis generate beyond conventional MIPS?", "expected_keywords": ["complex phase separation dynamics", "dynamic instabilities", "pattern-forming systems"], "category": "MIPS"},
    {"id": 59, "question": "What experimental systems might explore the instabilities predicted by the chemotactic MIPS theory?", "expected_keywords": ["synthetic active matter", "tunable velocities", "chemical dynamics"], "category": "MIPS"},
    {"id": 60, "question": "What other forms of taxis does the theoretical framework suggest could be studied?", "expected_keywords": ["chemorepulsion", "durotaxis", "electrotaxis", "phototaxis"], "category": "MIPS"},
    {"id": 61, "question": "What are the three interconnected components introduced by the Segment Anything project?", "expected_keywords": ["task", "model", "dataset", "Segment Anything"], "category": "SAM"},
    {"id": 62, "question": "What is the Segment Anything Model abbreviated as?", "expected_keywords": ["SAM", "Segment Anything Model"], "category": "SAM"},
    {"id": 63, "question": "What is the name of the dataset introduced with SAM?", "expected_keywords": ["SA-1B", "Segment Anything 1B", "dataset"], "category": "SAM"},
    {"id": 64, "question": "How many masks and images does SA-1B contain?", "expected_keywords": ["1 billion masks", "11M images", "SA-1B"], "category": "SAM"},
    {"id": 65, "question": "What does the promptable segmentation task require the model to output?", "expected_keywords": ["valid segmentation mask", "prompt", "image"], "category": "SAM"},
    {"id": 66, "question": "What can a segmentation prompt include in SAM?", "expected_keywords": ["points", "box", "mask", "text"], "category": "SAM"},
    {"id": 67, "question": "Why must SAM be ambiguity-aware?", "expected_keywords": ["ambiguous prompt", "multiple objects", "reasonable mask"], "category": "SAM"},
    {"id": 68, "question": "What are the three main parts of the SAM architecture?", "expected_keywords": ["image encoder", "prompt encoder", "mask decoder"], "category": "SAM"},
    {"id": 69, "question": "Why does SAM separate the image encoder from the prompt encoder and mask decoder?", "expected_keywords": ["reuse image embedding", "amortized", "different prompts"], "category": "SAM"},
    {"id": 70, "question": "How fast can the prompt encoder and mask decoder predict a mask in a web browser?", "expected_keywords": ["50ms", "web browser", "prompt encoder", "mask decoder"], "category": "SAM"},
    {"id": 71, "question": "What are the three stages of SAM's data engine?", "expected_keywords": ["assisted-manual", "semi-automatic", "fully automatic"], "category": "SAM"},
    {"id": 72, "question": "What happens in the assisted-manual stage of the SAM data engine?", "expected_keywords": ["SAM assists annotators", "annotating masks", "interactive segmentation"], "category": "SAM"},
    {"id": 73, "question": "What happens in the semi-automatic stage of the SAM data engine?", "expected_keywords": ["automatically generate masks", "likely object locations", "annotators"], "category": "SAM"},
    {"id": 74, "question": "What happens in the fully automatic stage of the SAM data engine?", "expected_keywords": ["regular grid", "foreground points", "100 masks per image"], "category": "SAM"},
    {"id": 75, "question": "What kinds of images does SA-1B use according to the paper?", "expected_keywords": ["licensed", "privacy-preserving", "diverse"], "category": "SAM"},
    {"id": 76, "question": "What responsible AI concern does the SAM paper evaluate?", "expected_keywords": ["fairness", "bias", "people", "groups"], "category": "SAM"},
    {"id": 77, "question": "Across which perceived attributes does the SAM paper evaluate segmentation performance?", "expected_keywords": ["gender presentation", "age group", "skin tone"], "category": "SAM"},
    {"id": 78, "question": "What zero-shot transfer tasks are evaluated for SAM?", "expected_keywords": ["edge detection", "object proposal generation", "instance segmentation", "text-to-mask"], "category": "SAM"},
    {"id": 79, "question": "What image encoder is used in SAM unless otherwise specified?", "expected_keywords": ["MAE", "ViT-H", "image encoder"], "category": "SAM"},
    {"id": 80, "question": "Under what license is SAM released according to the paper?", "expected_keywords": ["Apache 2.0", "permissive open license", "release"], "category": "SAM"},
    {"id": 81, "question": "What is the title of the governance paper?", "expected_keywords": ["Algorithmic Reflexive Governance", "Socio-Techno-Ecological Systems"], "category": "Governance"},
    {"id": 82, "question": "Who are the authors of the Algorithmic Reflexive Governance paper?", "expected_keywords": ["Jeremy Pitt", "John Dryzek", "Josiah Ober"], "category": "Governance"},
    {"id": 83, "question": "What workshop is mentioned on the first page of the governance paper?", "expected_keywords": ["Montreal", "Sept. 18-20, 2020", "Society Sustainability in the Digital Age"], "category": "Governance"},
    {"id": 84, "question": "What is reflexivity generally defined as in the governance paper?", "expected_keywords": ["ability", "reconfigure itself", "reflection", "performance"], "category": "Governance"},
    {"id": 85, "question": "What is the core aim of the governance paper?", "expected_keywords": ["self-governing", "socio-technical systems", "ecological impact", "algorithms"], "category": "Governance"},
    {"id": 86, "question": "What are the four dimensions of reflexivity discussed in the paper?", "expected_keywords": ["sources of knowledge", "public discourse", "institutional architecture", "institutional dynamics"], "category": "Governance"},
    {"id": 87, "question": "What tension defines the sources of knowledge dimension?", "expected_keywords": ["public participation", "expertise", "sources of knowledge"], "category": "Governance"},
    {"id": 88, "question": "What tension defines the composition of public discourse dimension?", "expected_keywords": ["diversity", "consensus", "public discourse"], "category": "Governance"},
    {"id": 89, "question": "What tension defines the institutional architecture dimension?", "expected_keywords": ["polycentricity", "centralization", "institutional architecture"], "category": "Governance"},
    {"id": 90, "question": "What tension defines the institutional dynamics dimension?", "expected_keywords": ["flexibility", "stability", "institutional dynamics"], "category": "Governance"},
    {"id": 91, "question": "What is Algorithmic Reflexive Governance according to the paper?", "expected_keywords": ["algorithms", "deliberative processes", "four dimensions of reflexivity"], "category": "Governance"},
    {"id": 92, "question": "What is Relevant Expertise Aggregation intended to resolve?", "expected_keywords": ["majority preference", "expert judgement", "REA"], "category": "Governance"},
    {"id": 93, "question": "How does Relevant Expertise Aggregation combine democratic and epistemic decision-making?", "expected_keywords": ["majority", "experts", "recommendations", "policy priorities"], "category": "Governance"},
    {"id": 94, "question": "What is the Zone of Dignity in the governance paper?", "expected_keywords": ["acceptable range", "policy options", "guardrails", "dignity"], "category": "Governance"},
    {"id": 95, "question": "What are the three components of the extended Zone of Dignity?", "expected_keywords": ["coordinate plane", "metrics", "meta-rules"], "category": "Governance"},
    {"id": 96, "question": "What does the paper say about hierarchical systems?", "expected_keywords": ["information flows up", "policies flow down", "centralized"], "category": "Governance"},
    {"id": 97, "question": "What does the paper say about polycentric systems?", "expected_keywords": ["multiple autonomous centers", "decision-making", "peer-peer"], "category": "Governance"},
    {"id": 98, "question": "What four layers of human involvement are described for community energy systems?", "expected_keywords": ["autonomous", "programmable", "interactive", "attentive"], "category": "Governance"},
    {"id": 99, "question": "What does the paper say Ashby's Design for a Brain implies about regulation?", "expected_keywords": ["regulator", "more complex", "system being regulated"], "category": "Governance"},
    {"id": 100, "question": "What advances does the governance paper say remain to be made?", "expected_keywords": ["integration", "common computational framework", "value-sensitive design"], "category": "Governance"},
]


def normalize_text(text):
    """
    文本预处理：去除换行、特殊字符、多余空格
    用于改进召回率计算的鲁棒性
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def check_recall(docs, expected_keywords):
    """
    改进版召回率计算:
    - 预处理文本，去除换行、特殊字符
    - 召回率 = 命中的关键词数 / 总关键词数
    """
    if not docs:
        return 0.0
    all_text = normalize_text(" ".join(docs))
    hits = 0
    for kw in expected_keywords:
        clean_kw = normalize_text(kw)
        if clean_kw and clean_kw in all_text:
            hits += 1
    return hits / len(expected_keywords)


def check_answer_quality(answer, expected_keywords):
    """
    改进版质量计算:
    - 预处理文本
    - 质量 = 命中的关键词数 / 总关键词数
    """
    if not answer:
        return 0.0
    answer_norm = normalize_text(answer)
    hits = 0
    for kw in expected_keywords:
        clean_kw = normalize_text(kw)
        if clean_kw and clean_kw in answer_norm:
            hits += 1
    return hits / len(expected_keywords)


def pairwise_judge(llm, question, answer_a, answer_b, expected_keywords, swap=False):
    """
    DPO风格对比评估:
    - 让LLM判断两个答案哪个更好
    - swap=True时交换位置消除偏见
    - 返回: "A", "B", 或 "TIE"
    """
    if swap:
        answer_a, answer_b = answer_b, answer_a
    
    prompt = f"""Compare two answers for the question. Which one is better?

Question: {question}
Expected concepts: {', '.join(expected_keywords)}

Answer A: {answer_a[:300]}
Answer B: {answer_b[:300]}

Criteria:
- Factual accuracy
- Completeness
- Relevance to expected concepts

Output ONLY one word: A, B, or TIE"""

    response = llm.generate(prompt, max_new_tokens=10).strip().upper()
    
    if "A" in response and "B" not in response:
        result = "A"
    elif "B" in response and "A" not in response:
        result = "B"
    else:
        result = "TIE"
    
    if swap:
        if result == "A":
            result = "B"
        elif result == "B":
            result = "A"
    
    return result


def load_knowledge(vector_store, bm25_store, data_dir="data"):
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return 0
    total_chunks = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            text_content = ""
            if file.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception as e:
                    continue
            elif file.endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_content += extracted + "\n"
                except Exception as e:
                    continue

            if text_content.strip():
                vec_count = vector_store.add_text(text_content)
                bm25_store.add(text_content)
                total_chunks += vec_count

    bm25_store.build()
    return total_chunks


def build_prompt(question, context):
    return f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""


def postprocess_response(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()


def retrieve_docs(vector_store, bm25_store, reranker, question, config_name, alpha=0.4):
    """
    根据配置检索文档
    """
    if config_name == "vector_only":
        docs = vector_store.search(question, k=3, threshold=0.3)
    elif config_name == "bm25_only":
        docs = bm25_store.search(question, top_k=3)
    elif config_name == "hybrid":
        vector_docs = vector_store.search(question, k=5, threshold=0.3)
        bm25_docs = bm25_store.search(question, top_k=5)
        scores = {}
        for rank, doc in enumerate(vector_docs, 1):
            scores[doc] = scores.get(doc, 0) + alpha * 1 / (60 + rank)
        for rank, doc in enumerate(bm25_docs, 1):
            scores[doc] = scores.get(doc, 0) + (1 - alpha) * 1 / (60 + rank)
        docs = [d for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:3]
    else:
        vector_docs = vector_store.search(question, k=5, threshold=0.3)
        bm25_docs = bm25_store.search(question, top_k=5)
        scores = {}
        for rank, doc in enumerate(vector_docs, 1):
            scores[doc] = scores.get(doc, 0) + alpha * 1 / (60 + rank)
        for rank, doc in enumerate(bm25_docs, 1):
            scores[doc] = scores.get(doc, 0) + (1 - alpha) * 1 / (60 + rank)
        merged = [d for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        if len(merged) > 3:
            docs = reranker.rerank(question, merged, 3)
        else:
            docs = merged[:3]
    
    return docs


def run_ablation(quick_test=False, test_n=10):
    """
    quick_test: 快速测试模式，只跑前 test_n 个问题
    test_n: 快速测试模式下的问题数量
    """
    alpha = 0.4
    
    test_cases = TEST_CASES[:test_n] if quick_test else TEST_CASES
    
    print("\n" + "=" * 70)
    print("Retrieval Ablation Study: Chunk Size + Retrieval Strategy")
    print(f"RRF Alpha: {alpha} (Vector) / {1-alpha} (BM25)")
    print("Evaluation: Improved Recall + DPO-style Pairwise Comparison")
    if quick_test:
        print(f"[QUICK TEST MODE] Testing {test_n} cases only")
    print("=" * 70)

    llm = LLM()
    embedding = Embedding()
    reranker = Rerank()

    chunk_sizes = [80, 200]
    configs = ["vector_only", "bm25_only", "hybrid", "hybrid_rerank"]
    
    all_results = {}
    all_badcase_log = []
    all_badcase_log.append("=" * 70)
    all_badcase_log.append("Retrieval Ablation Badcase Analysis")
    all_badcase_log.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_badcase_log.append("Evaluation Method: Improved Keyword Match + DPO-style Pairwise")
    if quick_test:
        all_badcase_log.append(f"[QUICK TEST MODE] {test_n} cases")
    all_badcase_log.append("=" * 70)

    pairwise_results = {
        "hybrid_vs_bm25": {"A_wins": 0, "B_wins": 0, "TIE": 0},
        "hybrid_rerank_vs_hybrid": {"A_wins": 0, "B_wins": 0, "TIE": 0},
    }

    for chunk_size in chunk_sizes:
        print(f"\n{'=' * 70}")
        print(f"Testing Chunk Size: {chunk_size}")
        print("=" * 70)

        vector_store = VectorStore(embedding, chunk_size=chunk_size)
        bm25_store = BM25Store(chunk_size=chunk_size)

        print("Loading knowledge base...")
        total_chunks = load_knowledge(vector_store, bm25_store)
        print(f"Loaded: {len(vector_store.texts)} chunks (Vector) + {len(bm25_store)} chunks (BM25)")

        all_results[chunk_size] = {}
        config_answers = {}

        for config in configs:
            print(f"  Testing {config}...")
            results = {"top1_hit": 0, "top3_hit": 0, "recall_sum": 0, "quality_sum": 0, "latency_sum": 0}
            config_answers[config] = {}
            
            config_badcase = []
            
            for case in test_cases:
                question = case["question"]
                expected = case["expected_keywords"]
                category = case.get("category", "Unknown")
                
                start_time = time.time()
                docs = retrieve_docs(vector_store, bm25_store, reranker, question, config, alpha)
                retrieve_time = time.time() - start_time
                
                top1_recall = check_recall(docs[:1], expected)
                top3_recall = check_recall(docs[:3], expected)
                
                if top1_recall > 0:
                    results["top1_hit"] += 1
                if top3_recall > 0:
                    results["top3_hit"] += 1
                
                results["recall_sum"] += top3_recall
                results["latency_sum"] += retrieve_time
                
                context = "\n".join(docs[:3]) if docs else "(empty)"
                prompt = build_prompt(question, context)
                output = llm.generate(prompt, max_new_tokens=128)
                answer = postprocess_response(output)
                quality = check_answer_quality(answer, expected)
                results["quality_sum"] += quality
                
                config_answers[config][case["id"]] = {
                    "answer": answer,
                    "docs": docs,
                    "recall": top3_recall,
                    "quality": quality
                }
                
                if top3_recall == 0:
                    config_badcase.append(f"\n--- Badcase #{case['id']} [{category}] ---")
                    config_badcase.append(f"Question: {question}")
                    config_badcase.append(f"Expected keywords: {expected}")
                    config_badcase.append(f"Retrieved docs ({len(docs)}): {[d[:50]+'...' for d in docs[:3]]}")
                    config_badcase.append(f"Top-3 Recall: {top3_recall:.2f}")
                    config_badcase.append(f"Answer quality: {quality:.2f}")
            
            all_results[chunk_size][config] = results
            
            if config_badcase:
                all_badcase_log.append(f"\n{'=' * 70}")
                all_badcase_log.append(f"Chunk Size: {chunk_size} | Config: {config}")
                all_badcase_log.append(f"Total Badcases: {len(config_badcase) // 6}")
                all_badcase_log.append("=" * 70)
                all_badcase_log.extend(config_badcase)

        print(f"\n  Running DPO-style Pairwise Comparison...")
        
        for case in test_cases:
            question = case["question"]
            expected = case["expected_keywords"]
            case_id = case["id"]
            
            answer_hybrid = config_answers["hybrid"][case_id]["answer"]
            answer_bm25 = config_answers["bm25_only"][case_id]["answer"]
            answer_rerank = config_answers["hybrid_rerank"][case_id]["answer"]
            
            result1 = pairwise_judge(llm, question, answer_hybrid, answer_bm25, expected, swap=False)
            result2 = pairwise_judge(llm, question, answer_hybrid, answer_bm25, expected, swap=True)
            
            if result1 == result2:
                final_result = result1
            else:
                final_result = "TIE"
            
            if final_result == "A":
                pairwise_results["hybrid_vs_bm25"]["A_wins"] += 1
            elif final_result == "B":
                pairwise_results["hybrid_vs_bm25"]["B_wins"] += 1
            else:
                pairwise_results["hybrid_vs_bm25"]["TIE"] += 1
            
            result1 = pairwise_judge(llm, question, answer_rerank, answer_hybrid, expected, swap=False)
            result2 = pairwise_judge(llm, question, answer_rerank, answer_hybrid, expected, swap=True)
            
            if result1 == result2:
                final_result = result1
            else:
                final_result = "TIE"
            
            if final_result == "A":
                pairwise_results["hybrid_rerank_vs_hybrid"]["A_wins"] += 1
            elif final_result == "B":
                pairwise_results["hybrid_rerank_vs_hybrid"]["B_wins"] += 1
            else:
                pairwise_results["hybrid_rerank_vs_hybrid"]["TIE"] += 1

    print("\n" + "=" * 70)
    print("Final Results Summary")
    print("=" * 70)

    total = len(test_cases)

    print(f"\n{'Chunk':<8} {'Config':<18} {'Top-1':<10} {'Top-3':<10} {'Recall':<10} {'Quality':<10}")
    print("-" * 74)

    for chunk_size in chunk_sizes:
        for config in configs:
            r = all_results[chunk_size][config]
            config_label = config.replace("_", " ").title()
            print(f"{chunk_size:<8} {config_label:<18} {r['top1_hit']:>2}/{total:<5} {r['top3_hit']:>2}/{total:<5} "
                  f"{r['recall_sum']/total*100:.1f}%{'':<5} {r['quality_sum']/total*100:.1f}%")
        print("-" * 74)

    print("\n" + "=" * 70)
    print("DPO-style Pairwise Comparison Results")
    print("=" * 70)
    
    print(f"\n{'Comparison':<30} {'A Wins':<12} {'B Wins':<12} {'TIE':<12} {'Win Rate':<12}")
    print("-" * 78)
    
    for comp_name, results in pairwise_results.items():
        a_wins = results["A_wins"]
        b_wins = results["B_wins"]
        ties = results["TIE"]
        total_decisive = a_wins + b_wins
        win_rate = a_wins / total_decisive * 100 if total_decisive > 0 else 0
        print(f"{comp_name:<30} {a_wins:>5}{'':<6} {b_wins:>5}{'':<6} {ties:>5}{'':<6} {win_rate:.1f}%")

    print("\n" + "=" * 70)
    print("Best Configuration Analysis")
    print("=" * 70)

    best_top3 = 0
    best_config = None
    for chunk_size in chunk_sizes:
        for config in configs:
            top3 = all_results[chunk_size][config]["top3_hit"]
            if top3 > best_top3:
                best_top3 = top3
                best_config = (chunk_size, config)

    print(f"\nBest Top-3: {best_top3}/{total} ({best_top3/total*100:.1f}%)")
    print(f"Configuration: chunk_size={best_config[0]}, strategy={best_config[1]}")

    print("\n" + "=" * 70)
    print("Chunk Size Impact Analysis")
    print("=" * 70)
    print(f"\n{'Chunk Size':<12} {'Vector':<12} {'BM25':<12} {'Hybrid':<12} {'Hybrid+Rerank':<12}")
    print("-" * 60)
    for chunk_size in chunk_sizes:
        vec = all_results[chunk_size]["vector_only"]["top3_hit"]
        bm25 = all_results[chunk_size]["bm25_only"]["top3_hit"]
        hybrid = all_results[chunk_size]["hybrid"]["top3_hit"]
        hybrid_rerank = all_results[chunk_size]["hybrid_rerank"]["top3_hit"]
        print(f"{chunk_size:<12} {vec:>2}/{total:<8} {bm25:>2}/{total:<8} {hybrid:>2}/{total:<8} {hybrid_rerank:>2}/{total:<8}")

    all_badcase_log.append(f"\n\n{'=' * 70}")
    all_badcase_log.append("DPO-style Pairwise Comparison Results")
    all_badcase_log.append("=" * 70)
    for comp_name, results in pairwise_results.items():
        a_wins = results["A_wins"]
        b_wins = results["B_wins"]
        ties = results["TIE"]
        total_decisive = a_wins + b_wins
        win_rate = a_wins / total_decisive * 100 if total_decisive > 0 else 0
        all_badcase_log.append(f"\n{comp_name}:")
        all_badcase_log.append(f"  A Wins: {a_wins} | B Wins: {b_wins} | TIE: {ties}")
        all_badcase_log.append(f"  Win Rate: {win_rate:.1f}%")

    log_path = "eval/results/rag_badcase_log.txt"
    os.makedirs("eval/results", exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_badcase_log))
    print(f"\nBadcase日志已保存到: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Retrieval Ablation Study")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--n", type=int, default=10, help="快速测试模式下的问题数量")
    args = parser.parse_args()
    
    run_ablation(quick_test=args.quick, test_n=args.n)
