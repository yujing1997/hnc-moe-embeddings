#!/usr/bin/env python3
"""
01_generate_embeddings.py — Dual-Embedding Generation for 5 Models
===================================================================
Generates dense text embeddings for HNC pathology and consultation reports
using five language models:
  B1: SMB-v1-1.7B   (standardmodelbio/smb-v1-1.7b, 2048-dim, last-token pooling)
  B2: Clinical-Longformer (yikuan8/Clinical-Longformer, 768-dim, mean pooling)
  B3: Llama 3.3 70B (llama3.3:latest via Ollama /api/embed, 8192-dim)
  B4: Nomic-Embed-Text v1.5 (nomic-embed-text:latest via Ollama /api/embed, 768-dim)
  B5: SMB-v1-8B     (standardmodelbio/smb-v1-8b, 4096-dim, last-token pooling, Qwen3-8B base)
  B6: SMB-v1-Qwen3-8B-MO (standardmodelbio/SMB-v1_Qwen3-8b_multi-objective, 4096-dim, unstructured text)
  B7: SMB-v1-Llama3-8B-CL (standardmodelbio/SMB-v1_Llama3-8b_curriculum, 4096-dim, unstructured text)

Strategy: Dual-embedding (pathology + consultation embedded separately, concatenated).
Missing modalities are zero-padded.

Output:
  {output_dir}/{model_key}/{model_key}_dual_embeddings.pkl — dict: patient_id (str) → numpy array
  {output_dir}/{model_key}/{model_key}_metadata.csv — per-patient metadata (has_path, has_consult, etc.)

Usage:
  python 01_generate_embeddings.py --model all
  python 01_generate_embeddings.py --model longformer
  python 01_generate_embeddings.py --model smb
  python 01_generate_embeddings.py --model smb8b
  python 01_generate_embeddings.py --model llama
  python 01_generate_embeddings.py --model nomic

Author: Yujing Zou - Embeddings vs. Extraction comparison study
"""

import os
# Redirect HuggingFace cache to /Data/ (1.7 TB free) instead of ~/.cache (96 GB free)
os.environ["HF_HOME"] = "/Data/Yujing/hf_cache"

import sys
import argparse
import pickle
import time
import logging
import json
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========================== CONFIGURATION ==========================

# Report directories
CONSULT_DIR = Path("/Data/Yujing/HNC_OutcomePred/HNC_Reports/consultation_notes")
PATHOLOGY_DIR = Path("/Data/Yujing/HNC_OutcomePred/HNC_Reports/pathology_reports")

# Default output
DEFAULT_OUTPUT_DIR = Path("/Data/Yujing/HNC_OutcomePred/Reports_Agents/Revision1/Downstream_Analysis/moe_embeddings_comp/embeddings")

# Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# Model configurations
MODEL_CONFIGS = {
    "longformer": {
        "hf_name": "yikuan8/Clinical-Longformer",
        "embedding_dim": 768,
        "max_tokens": 4096,
        "pooling": "mean",
        "description": "Clinical-Longformer (768-dim, mean pooling, MIMIC-III pretrained)",
    },
    "smb": {
        "hf_name": "standardmodelbio/smb-v1-1.7b",
        "embedding_dim": 2048,
        "max_tokens": 4096,
        "pooling": "last_token",
        "description": "SMB-v1-1.7B (2048-dim, last-token pooling, biomedical)",
    },
    "llama": {
        "ollama_model": "llama3.3:latest",
        "embedding_dim": 8192,
        "max_tokens": 131072,
        "pooling": "ollama_api",
        "description": "Llama 3.3 70B (8192-dim, decoder, Ollama /api/embed)",
    },
    "nomic": {
        "ollama_model": "nomic-embed-text:latest",
        "embedding_dim": 768,
        "max_tokens": 8192,
        "pooling": "ollama_api",
        "description": "Nomic-Embed-Text v1.5 (768-dim, encoder, contrastive-trained, Ollama /api/embed)",
    },
    "smb8b": {
        "hf_name": "standardmodelbio/smb-v1-8b",
        "embedding_dim": 4096,
        "max_tokens": 4096,
        "pooling": "last_token",
        "description": "SMB-v1-8B (4096-dim, last-token pooling, Qwen3-8B base, biomedical SFT+JEPA)",
    },
    "smb8b_qwen3_mo": {
        "hf_name": "standardmodelbio/SMB-v1_Qwen3-8b_multi-objective",
        "embedding_dim": 4096,
        "max_tokens": 3300,
        "pooling": "last_token",
        "description": "SMB-v1-Qwen3-8B-MO (4096-dim, Qwen3-8B base, SFT+JEPA multi-objective, unstructured text)",
    },
    "smb8b_llama3_cl": {
        "hf_name": "standardmodelbio/SMB-v1_Llama3-8b_curriculum",
        "embedding_dim": 4096,
        "max_tokens": 8192,
        "pooling": "last_token",
        "description": "SMB-v1-Llama3-8B-CL (4096-dim, Llama3-8B base, SFT+JEPA curriculum learning, unstructured text)",
    },
}

# ========================== CHUNKING CONFIGURATION ==========================
# Sliding-window overlap (HONeYBEE-inspired, Ref: D样 et al., npj Digital Medicine)
# to handle texts exceeding model context windows.  Chunks are embedded
# independently and mean-pooled.  Only activates when text exceeds max_tokens;
# costs zero runtime overhead when all texts fit within the context window.
CHUNK_OVERLAP_FRACTION = 0.10  # 10% overlap between adjacent windows


def chunk_text_by_words(text: str, max_words: int,
                        overlap_fraction: float = CHUNK_OVERLAP_FRACTION) -> List[str]:
    """Split *text* into overlapping chunks by word count.

    Used for Ollama models where we lack a local tokenizer and must
    estimate token count from word count (word_count × 1.3).
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    overlap_words = max(1, int(max_words * overlap_fraction))
    stride = max_words - overlap_words
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += stride
    return chunks


# ========================== REPORT LOADING ==========================

def load_reports() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load all pathology and consultation reports.
    
    Returns:
        path_reports: dict patient_id -> text
        consult_reports: dict patient_id -> text
    """
    path_reports = {}
    consult_reports = {}
    
    if PATHOLOGY_DIR.exists():
        for f in sorted(PATHOLOGY_DIR.glob("*.txt")):
            pid = f.stem
            text = f.read_text(encoding='utf-8', errors='replace').strip()
            if text:
                path_reports[pid] = text
    
    if CONSULT_DIR.exists():
        for f in sorted(CONSULT_DIR.glob("*.txt")):
            pid = f.stem
            text = f.read_text(encoding='utf-8', errors='replace').strip()
            if text:
                consult_reports[pid] = text
    
    logger.info(f"Loaded {len(path_reports)} pathology reports, {len(consult_reports)} consultation reports")
    
    all_pids = sorted(set(path_reports.keys()) | set(consult_reports.keys()))
    both = sum(1 for p in all_pids if p in path_reports and p in consult_reports)
    path_only = sum(1 for p in all_pids if p in path_reports and p not in consult_reports)
    consult_only = sum(1 for p in all_pids if p not in path_reports and p in consult_reports)
    logger.info(f"Total unique patients: {len(all_pids)} "
                f"(both: {both}, path_only: {path_only}, consult_only: {consult_only})")
    
    return path_reports, consult_reports


# ========================== HUGGINGFACE EMBEDDERS ==========================

class HFEmbedder:
    """Base embedder for HuggingFace transformer models.

    Supports sliding-window chunking with mean-pooling for texts
    exceeding the model's context window (overlap controlled by
    CHUNK_OVERLAP_FRACTION).  Attribute ``_last_n_chunks`` exposes how
    many chunks were used for the most recent ``embed_text`` call.
    """

    def __init__(self, model_name: str, max_tokens: int, pooling: str, device: str = None):
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._last_n_chunks = 1  # track chunks from last embed_text call

        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get actual embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Loaded {model_name}: dim={self.embedding_dim}, max_tokens={max_tokens}, pooling={pooling}")

    # ---- low-level: embed a single chunk that fits in the context window ----
    @torch.no_grad()
    def _embed_chunk(self, text: str) -> np.ndarray:
        """Embed a single text chunk. Returns 1-D numpy array."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        if self.pooling == "mean":
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked = hidden_states * attention_mask
            embedding = masked.sum(dim=1) / attention_mask.sum(dim=1)
        elif self.pooling == "last_token":
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            embedding = hidden_states[0, seq_lengths[0], :].unsqueeze(0)
        elif self.pooling == "cls":
            embedding = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return embedding.cpu().numpy().flatten().astype(np.float32)

    # ---- public: embed with automatic sliding-window chunking ----
    def embed_text(self, text: str) -> np.ndarray:
        """Embed *text*, chunking with a sliding window if it exceeds max_tokens.

        When the text fits within the context window, a single forward pass
        is used (no overhead).  Otherwise the text is split into overlapping
        token-level chunks, each chunk is embedded independently, and the
        chunk embeddings are **mean-pooled** to produce a single vector.
        """
        if not text or not text.strip():
            self._last_n_chunks = 0
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Tokenize without special tokens to count raw length
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        effective_max = self.max_tokens - 2  # reserve room for [CLS]/[SEP] or BOS/EOS

        if len(token_ids) <= effective_max:
            self._last_n_chunks = 1
            return self._embed_chunk(text)

        # --- sliding-window chunking ---
        overlap = max(1, int(effective_max * CHUNK_OVERLAP_FRACTION))
        stride = effective_max - overlap
        chunks_text: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + effective_max, len(token_ids))
            chunk_text = self.tokenizer.decode(
                token_ids[start:end], skip_special_tokens=True
            )
            chunks_text.append(chunk_text)
            if end >= len(token_ids):
                break
            start += stride

        self._last_n_chunks = len(chunks_text)
        logger.info(
            f"Chunked text into {len(chunks_text)} chunks "
            f"({len(token_ids)} tokens, window={effective_max}, overlap={overlap})"
        )

        chunk_embs = np.stack([self._embed_chunk(c) for c in chunks_text])
        return chunk_embs.mean(axis=0).astype(np.float32)


class SMBEmbedder:
    """SMB-v1-1.7B embedder following official docs (https://docs.standardmodel.bio/your-own-data).

    Key differences from generic HFEmbedder:
      - Uses AutoModelForCausalLM (not AutoModel) per official docs
      - Extracts hidden_states[-1][:, -1, :] (not last_hidden_state) per official docs
      - Wraps raw clinical text in <notes>...</notes> tags
      - Uses last-token pooling for causal inference

    Supports sliding-window chunking for texts exceeding max_tokens.
    """

    def __init__(self, model_name: str, max_tokens: int, pooling: str, device: str = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._last_n_chunks = 1

        logger.info(f"Loading {model_name} (AutoModelForCausalLM) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        # Count tokens consumed by <notes>...</notes> wrapper
        self._wrapper_tokens = len(
            self.tokenizer.encode("<notes></notes>", add_special_tokens=False)
        )
        logger.info(
            f"Loaded {model_name}: dim={self.embedding_dim}, max_tokens={max_tokens}, "
            f"pooling={pooling}, wrapper overhead={self._wrapper_tokens} tokens"
        )

    @torch.no_grad()
    def _embed_chunk(self, text: str) -> np.ndarray:
        """Embed a single <notes>-wrapped chunk."""
        wrapped = f"<notes>{text}</notes>"
        inputs = self.tokenizer(
            wrapped,
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
        ).to(self.model.device)

        outputs = self.model(inputs.input_ids, output_hidden_states=True)
        vec = outputs.hidden_states[-1][:, -1, :]  # (1, hidden_dim)
        return vec.cpu().float().numpy().flatten().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with automatic sliding-window chunking if needed."""
        if not text or not text.strip():
            self._last_n_chunks = 0
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Count tokens for raw text (excluding <notes> wrapper)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        effective_max = self.max_tokens - self._wrapper_tokens - 1  # wrapper + BOS

        if len(token_ids) <= effective_max:
            self._last_n_chunks = 1
            return self._embed_chunk(text)

        # --- sliding-window chunking ---
        overlap = max(1, int(effective_max * CHUNK_OVERLAP_FRACTION))
        stride = effective_max - overlap
        chunks_text: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + effective_max, len(token_ids))
            chunk_text = self.tokenizer.decode(
                token_ids[start:end], skip_special_tokens=True
            )
            chunks_text.append(chunk_text)
            if end >= len(token_ids):
                break
            start += stride

        self._last_n_chunks = len(chunks_text)
        logger.info(
            f"SMB chunked: {len(token_ids)} tokens -> {len(chunks_text)} chunks "
            f"(window={effective_max}, overlap={overlap})"
        )

        chunk_embs = np.stack([self._embed_chunk(c) for c in chunks_text])
        return chunk_embs.mean(axis=0).astype(np.float32)


class SMBUnstructuredEmbedder:
    """Embedder for SMB-v1 unstructured-text models (SMBUnstructuredForCausalLM).

    Unlike the original SMBEmbedder (for structured EHR data wrapped in <notes> tags),
    these models are purpose-built for unstructured clinical text via their custom
    SMBUnstructuredForCausalLM architecture.  They take raw text directly,
    no <notes> wrapping needed.

    Key differences from SMBEmbedder:
      - Uses AutoModelForCausalLM with trust_remote_code=True to load custom architecture
      - No <notes>...</notes> wrapping (model accepts raw text)
      - Same embedding extraction: hidden_states[-1][:, -1, :] (last-token pooling)

    Supported models:
      - standardmodelbio/SMB-v1_Qwen3-8b_multi-objective  (Qwen3-8B base, multi-objective SFT+JEPA)
      - standardmodelbio/SMB-v1_Llama3-8b_curriculum       (Llama3-8B base, curriculum SFT+JEPA)
    """

    def __init__(self, model_name: str, max_tokens: int, pooling: str, device: str = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._last_n_chunks = 1

        logger.info(f"Loading {model_name} (SMBUnstructuredForCausalLM) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        logger.info(
            f"Loaded {model_name}: dim={self.embedding_dim}, max_tokens={max_tokens}, "
            f"pooling={pooling}"
        )

    @torch.no_grad()
    def _embed_chunk(self, text: str) -> np.ndarray:
        """Embed a single text chunk (raw text, no wrapping)."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
        ).to(self.model.device)

        outputs = self.model(
            input_ids=inputs.input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        vec = outputs.hidden_states[-1][:, -1, :]  # (1, hidden_dim)
        return vec.cpu().float().numpy().flatten().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with automatic sliding-window chunking if needed."""
        if not text or not text.strip():
            self._last_n_chunks = 0
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Count tokens for raw text
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        effective_max = self.max_tokens - 2  # reserve room for special tokens

        if len(token_ids) <= effective_max:
            self._last_n_chunks = 1
            return self._embed_chunk(text)

        # --- sliding-window chunking ---
        overlap = max(1, int(effective_max * CHUNK_OVERLAP_FRACTION))
        stride = effective_max - overlap
        chunks_text: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + effective_max, len(token_ids))
            chunk_text = self.tokenizer.decode(
                token_ids[start:end], skip_special_tokens=True
            )
            chunks_text.append(chunk_text)
            if end >= len(token_ids):
                break
            start += stride

        self._last_n_chunks = len(chunks_text)
        logger.info(
            f"SMBUnstructured chunked: {len(token_ids)} tokens -> {len(chunks_text)} chunks "
            f"(window={effective_max}, overlap={overlap})"
        )

        chunk_embs = np.stack([self._embed_chunk(c) for c in chunks_text])
        return chunk_embs.mean(axis=0).astype(np.float32)


# ========================== OLLAMA EMBEDDER ==========================

class OllamaEmbedder:
    """Embedder using Ollama /api/embed endpoint.

    Supports sliding-window chunking (by word count) for texts
    exceeding max_tokens.  Token count is estimated as word_count × 1.3.
    """

    _WORDS_PER_TOKEN = 1.3  # conservative estimate for clinical text

    def __init__(self, model_name: str = "llama3.3:latest",
                 max_tokens: int = 131072,
                 base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embed"
        self.max_tokens = max_tokens
        self._last_n_chunks = 1

        # Probe embedding dimension
        logger.info(f"Probing Ollama embedding dimension for {model_name}...")
        test_emb = self._embed_single("test")
        self.embedding_dim = len(test_emb)
        logger.info(f"Ollama {model_name}: dim={self.embedding_dim}, max_tokens={max_tokens}")

    def _embed_single(self, text: str, max_retries: int = 3) -> np.ndarray:
        """Embed text via Ollama API with retries."""
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.embed_url,
                    json={"model": self.model_name, "input": text},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return np.array(data["embeddings"][0], dtype=np.float32)
                elif "embedding" in data:
                    return np.array(data["embedding"], dtype=np.float32)
                else:
                    logger.warning(f"Unexpected Ollama response keys: {data.keys()}")
                    return np.zeros(
                        self.embedding_dim if hasattr(self, "embedding_dim") else 8192,
                        dtype=np.float32,
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Ollama failed after {max_retries} attempts: {e}")
                    return np.zeros(
                        self.embedding_dim if hasattr(self, "embedding_dim") else 8192,
                        dtype=np.float32,
                    )

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with automatic word-based chunking if it exceeds max_tokens."""
        if not text or not text.strip():
            self._last_n_chunks = 0
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Estimate token count
        n_words = len(text.split())
        est_tokens = int(n_words * self._WORDS_PER_TOKEN)

        if est_tokens <= self.max_tokens:
            self._last_n_chunks = 1
            return self._embed_single(text)

        # --- word-based sliding-window chunking ---
        max_words = int(self.max_tokens / self._WORDS_PER_TOKEN)
        chunks = chunk_text_by_words(text, max_words, CHUNK_OVERLAP_FRACTION)
        self._last_n_chunks = len(chunks)
        logger.info(
            f"Ollama chunked: ~{est_tokens} est. tokens -> {len(chunks)} chunks "
            f"(max_words={max_words})"
        )

        chunk_embs = np.stack([self._embed_single(c) for c in chunks])
        return chunk_embs.mean(axis=0).astype(np.float32)


# ========================== DUAL EMBEDDING ==========================

def generate_dual_embeddings(
    embedder,
    path_reports: Dict[str, str],
    consult_reports: Dict[str, str],
    patient_ids: List[str],
    model_key: str,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Generate dual embeddings (pathology + consultation) for all patients.
    
    For each patient:
      - Embed pathology report (or zeros if missing)
      - Embed consultation report (or zeros if missing)
      - Concatenate: [path_emb | consult_emb]
    
    Returns:
        embeddings: dict patient_id -> concatenated numpy array
        metadata: DataFrame with per-patient info
    """
    single_dim = embedder.embedding_dim
    concat_dim = single_dim * 2
    
    embeddings = {}
    metadata_rows = []
    
    logger.info(f"Generating dual embeddings for {len(patient_ids)} patients "
                f"(model={model_key}, single_dim={single_dim}, concat_dim={concat_dim})")
    
    for pid in tqdm(patient_ids, desc=f"Embedding ({model_key})", ncols=100):
        has_path = pid in path_reports
        has_consult = pid in consult_reports
        
        # Embed pathology
        if has_path:
            path_text = path_reports[pid]
            path_emb = embedder.embed_text(path_text)
            path_tokens = len(path_text.split())  # approximate word count
            path_n_chunks = getattr(embedder, '_last_n_chunks', 1)
        else:
            path_emb = np.zeros(single_dim, dtype=np.float32)
            path_tokens = 0
            path_n_chunks = 0

        # Embed consultation
        if has_consult:
            consult_text = consult_reports[pid]
            consult_emb = embedder.embed_text(consult_text)
            consult_tokens = len(consult_text.split())
            consult_n_chunks = getattr(embedder, '_last_n_chunks', 1)
        else:
            consult_emb = np.zeros(single_dim, dtype=np.float32)
            consult_tokens = 0
            consult_n_chunks = 0

        # Concatenate [path | consult]
        concat_emb = np.concatenate([path_emb, consult_emb])
        assert concat_emb.shape == (concat_dim,), f"Shape mismatch: {concat_emb.shape} vs ({concat_dim},)"

        embeddings[pid] = concat_emb

        metadata_rows.append({
            "patient_id": pid,
            "has_pathology": has_path,
            "has_consultation": has_consult,
            "has_both": has_path and has_consult,
            "path_word_count": path_tokens,
            "consult_word_count": consult_tokens,
            "path_n_chunks": path_n_chunks,
            "consult_n_chunks": consult_n_chunks,
        })

    metadata = pd.DataFrame(metadata_rows)

    # --- Chunking summary ---
    chunked_path = sum(1 for r in metadata_rows if r["path_n_chunks"] > 1)
    chunked_consult = sum(1 for r in metadata_rows if r["consult_n_chunks"] > 1)
    if chunked_path > 0 or chunked_consult > 0:
        logger.info(
            f"Chunking summary: {chunked_path} pathology, {chunked_consult} consultation "
            f"reports required sliding-window chunking"
        )
    else:
        logger.info("No reports required chunking (all fit within context window)")

    return embeddings, metadata


# ========================== LLAMA COMBINED EMBEDDING ==========================

def generate_llama_combined_embeddings(
    embedder: OllamaEmbedder,
    path_reports: Dict[str, str],
    consult_reports: Dict[str, str],
    patient_ids: List[str],
) -> Dict[str, np.ndarray]:
    """
    For Llama 3.3 70B (131K context), also generate single combined embeddings
    where path + consult are concatenated as text before embedding.
    This is only feasible for Llama due to its large context window.
    
    Returns:
        embeddings: dict patient_id -> single embedding vector (4096-dim)
    """
    embeddings = {}
    
    logger.info(f"Generating combined (single-pass) Llama embeddings for {len(patient_ids)} patients")
    
    for pid in tqdm(patient_ids, desc="Llama combined", ncols=100):
        parts = []
        if pid in path_reports:
            parts.append(f"=== PATHOLOGY REPORT ===\n{path_reports[pid]}")
        if pid in consult_reports:
            parts.append(f"=== CONSULTATION REPORT ===\n{consult_reports[pid]}")
        
        combined_text = "\n\n".join(parts) if parts else ""
        
        if combined_text:
            emb = embedder.embed_text(combined_text)
        else:
            emb = np.zeros(embedder.embedding_dim, dtype=np.float32)
        
        embeddings[pid] = emb
    
    return embeddings


def generate_combined_embeddings(
    embedder,
    path_reports: Dict[str, str],
    consult_reports: Dict[str, str],
    patient_ids: List[str],
    model_key: str,
) -> Dict[str, np.ndarray]:
    """
    Generate single combined embeddings for models with large context windows.
    Path + consult text is concatenated before embedding.
    
    Returns:
        embeddings: dict patient_id -> single embedding vector
    """
    embeddings = {}
    
    logger.info(f"Generating combined (single-pass) {model_key} embeddings for {len(patient_ids)} patients")
    
    for pid in tqdm(patient_ids, desc=f"{model_key} combined", ncols=100):
        parts = []
        if pid in path_reports:
            parts.append(f"=== PATHOLOGY REPORT ===\n{path_reports[pid]}")
        if pid in consult_reports:
            parts.append(f"=== CONSULTATION REPORT ===\n{consult_reports[pid]}")
        
        combined_text = "\n\n".join(parts) if parts else ""
        
        if combined_text:
            emb = embedder.embed_text(combined_text)
        else:
            emb = np.zeros(embedder.embedding_dim, dtype=np.float32)
        
        embeddings[pid] = emb
    
    return embeddings


# ========================== MAIN ==========================

def create_embedder(model_key: str):
    """Factory function to create the appropriate embedder."""
    config = MODEL_CONFIGS[model_key]
    
    if model_key == "longformer":
        return HFEmbedder(
            model_name=config["hf_name"],
            max_tokens=config["max_tokens"],
            pooling=config["pooling"],
        )
    elif model_key in ("smb", "smb8b"):
        return SMBEmbedder(
            model_name=config["hf_name"],
            max_tokens=config["max_tokens"],
            pooling=config["pooling"],
        )
    elif model_key in ("smb8b_qwen3_mo", "smb8b_llama3_cl"):
        return SMBUnstructuredEmbedder(
            model_name=config["hf_name"],
            max_tokens=config["max_tokens"],
            pooling=config["pooling"],
        )
    elif model_key in ("llama", "nomic"):
        return OllamaEmbedder(
            model_name=config["ollama_model"],
            max_tokens=config["max_tokens"],
        )
    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def run_model(model_key: str, path_reports: Dict, consult_reports: Dict, output_dir: Path):
    """Generate embeddings for one model and save."""
    
    all_pids = sorted(set(path_reports.keys()) | set(consult_reports.keys()))
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {MODEL_CONFIGS[model_key]['description']}")
    logger.info(f"Patients: {len(all_pids)}")
    logger.info(f"{'='*60}")
    
    # Create model-specific subfolder
    model_output_dir = output_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already done
    out_path = model_output_dir / f"{model_key}_dual_embeddings.pkl"
    if out_path.exists():
        logger.info(f"Already exists: {out_path}. Skipping. (Delete to regenerate.)")
        return
    
    start_time = time.time()
    
    # Create embedder
    embedder = create_embedder(model_key)
    
    # Generate dual embeddings
    embeddings, metadata = generate_dual_embeddings(
        embedder, path_reports, consult_reports, all_pids, model_key
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Dual embeddings generated in {elapsed:.1f}s "
                f"({elapsed/len(all_pids):.2f}s per patient)")
    
    # Save embeddings
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    
    # Save metadata
    meta_path = model_output_dir / f"{model_key}_metadata.csv"
    metadata.to_csv(meta_path, index=False)
    logger.info(f"Saved: {meta_path}")
    
    # For Llama, also generate combined (single-pass) embeddings
    if model_key == "llama":
        combined_out = model_output_dir / "llama_combined_embeddings.pkl"
        if not combined_out.exists():
            combined_embs = generate_llama_combined_embeddings(
                embedder, path_reports, consult_reports, all_pids
            )
            with open(combined_out, 'wb') as f:
                pickle.dump(combined_embs, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved combined Llama embeddings: {combined_out}")
    
    # Free GPU memory
    if hasattr(embedder, 'model'):
        del embedder.model
        del embedder.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del embedder
    
    # Print sample stats
    dims = [v.shape[0] for v in embeddings.values()]
    norms = [np.linalg.norm(v) for v in embeddings.values()]
    logger.info(f"Embedding dimension: {dims[0]}")
    logger.info(f"L2 norm — mean: {np.mean(norms):.2f}, min: {np.min(norms):.2f}, max: {np.max(norms):.2f}")


def main():
    global CONSULT_DIR, PATHOLOGY_DIR
    
    parser = argparse.ArgumentParser(description="Generate dual embeddings for HNC reports")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "longformer", "smb", "smb8b", "llama", "nomic",
                                 "smb8b_qwen3_mo", "smb8b_llama3_cl"],
                        help="Which model(s) to generate embeddings for")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for embeddings")
    parser.add_argument("--consult_dir", type=str, default=str(CONSULT_DIR),
                        help="Directory containing consultation notes")
    parser.add_argument("--pathology_dir", type=str, default=str(PATHOLOGY_DIR),
                        help="Directory containing pathology reports")
    args = parser.parse_args()
    
    # Update paths if provided
    CONSULT_DIR = Path(args.consult_dir)
    PATHOLOGY_DIR = Path(args.pathology_dir)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reports
    path_reports, consult_reports = load_reports()
    
    if not path_reports and not consult_reports:
        logger.error("No reports found. Check paths.")
        sys.exit(1)
    
    # Determine which models to run
    if args.model == "all":
        models = ["longformer", "smb", "smb8b", "llama", "nomic",
                  "smb8b_qwen3_mo", "smb8b_llama3_cl"]
    else:
        models = [args.model]
    
    # Run each model
    for model_key in models:
        try:
            run_model(model_key, path_reports, consult_reports, output_dir)
        except Exception as e:
            logger.error(f"Failed for model {model_key}: {e}", exc_info=True)
            continue
    
    logger.info("\nDone! All embeddings saved to: " + str(output_dir))


if __name__ == "__main__":
    main()
