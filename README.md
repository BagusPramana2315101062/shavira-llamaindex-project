# SHAVIRA Retrieval Experiment dengan LlamaIndex

Project ini dibuat untuk eksperimen proposal: perbandingan **BM25**, **dense retrieval BGE-M3 + FAISS**, dan **hybrid retrieval BM25 + FAISS via RRF** terhadap relevansi hasil _knowledge search_ SHAVIRA.

## 1. Struktur Folder

```text
shavira_llamaindex_project/
├─ data/
│  └─ raw/
│     ├─ jdih_undiksha_ac_id.jsonl
│     ├─ upttik_undiksha_ac_id.jsonl
│     ├─ undiksha_ac_id_pmb.jsonl
│     ├─ undiksha_ac_id_tentang_undiksha.jsonl
│     └─ Dataset_Validasi_Retrieval_SHAVIRA.xlsx
├─ results/
├─ storage/
├─ src/
│  ├─ check_dataset.py
│  └─ evaluate_retrieval.py
├─ requirements.txt
└─ README.md
```

## 2. Instalasi

Disarankan memakai Python 3.10 atau 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Catatan:

- Model `BAAI/bge-m3` akan diunduh otomatis dari Hugging Face saat pertama kali dijalankan.
- Ukuran model cukup besar, jadi proses pertama bisa lama.
- Jika `faiss-cpu` bermasalah di Windows, gunakan Anaconda/Miniconda atau WSL.

## 3. Menaruh Dataset

Letakkan lima file berikut ke folder `data/raw/`:

1. `jdih_undiksha_ac_id.jsonl`
2. `upttik_undiksha_ac_id.jsonl`
3. `undiksha_ac_id_pmb.jsonl`
4. `undiksha_ac_id_tentang_undiksha.jsonl`
5. `Dataset_Validasi_Retrieval_SHAVIRA.xlsx`

## 4. Cek Dataset

Jalankan:

```bash
python src/check_dataset.py --data-dir data/raw
```

Output yang diharapkan:

- JSONL memiliki field `text` dan `metadata`.
- Dataset validasi memiliki kolom `ID`, `Context`, `Question`, dan `Answer`.
- `Context exact record match` idealnya mendekati atau sama dengan `499/499`.

## 5. Jalankan Eksperimen Lengkap

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results
```

Konfigurasi bawaan:

- `chunk_size = 512`
- `chunk_overlap = 64`
- `eval_k = 5 10`
- `candidate_k = 50`
- `rrf_k = 10`
- `embedding_model = BAAI/bge-m3`
- `faiss_index = IndexFlatIP`

## 6. Uji Cepat Dulu

Untuk memastikan pipeline berjalan tanpa menunggu semua 499 query:

```bash
python src/evaluate_retrieval.py --data-dir data/raw --output-dir results --limit 10
```

## 7. Output Eksperimen

Setelah eksperimen selesai, folder `results/` akan berisi:

1. `summary_metrics.csv`
   - Ringkasan performa tiap metode pada K=5 dan K=10.

2. `per_query_metrics.csv`
   - Nilai metrik per query, per metode, per K.

3. `retrieval_details_top10.csv`
   - Detail hasil Top-10, termasuk node ID, skor, judul, URL, dan cuplikan Top-1.

4. `summary_and_details.xlsx`
   - Gabungan semua output dalam format Excel.

## 8. Definisi Evaluasi

Dataset validasi memiliki satu `Context` utama untuk setiap `Question`. Karena `Context` pada dataset validasi cocok dengan record JSONL, project ini mengevaluasi relevansi pada level `text_hash` record asal.

Artinya:

- Hasil retrieval tetap berupa chunk.
- Setiap chunk membawa metadata `text_hash` dari record asal.
- Hasil dianggap relevan jika `text_hash` chunk hasil retrieval sama dengan `text_hash` dari `Context` validasi.
- Hasil Top-K dideduplikasi berdasarkan `text_hash` agar satu record yang terpecah menjadi beberapa chunk tidak dihitung berkali-kali.

## 9. Perintah dengan Parameter Eksplisit

```bash
python src/evaluate_retrieval.py \
  --data-dir data/raw \
  --output-dir results \
  --chunk-size 512 \
  --chunk-overlap 64 \
  --eval-k 5 10 \
  --candidate-k 50 \
  --rrf-k 10 \
  --model-name BAAI/bge-m3
```

## 10. Interpretasi Awal

- Jika `Hit@5` atau `Hit@10` tinggi, gold context sering berhasil ditemukan dalam hasil Top-K.
- Jika `MRR` tinggi, gold context sering muncul pada peringkat awal.
- `Precision@K` dan `Recall@K` tetap disajikan sebagai metrik pendukung.
- `MAP` tidak digunakan karena pada dataset dengan satu gold context utama per unit evaluasi, nilainya cenderung setara dengan MRR.
- Jika `HYBRID_RRF` lebih tinggi dari BM25 dan FAISS, berarti penggabungan lexical dan semantic retrieval lebih sesuai untuk korpus SHAVIRA.
- Jika BM25 lebih tinggi, kemungkinan query validasi banyak memakai istilah yang sama dengan dokumen.
- Jika FAISS lebih tinggi, kemungkinan query validasi banyak membutuhkan pencocokan makna/parafrasa.
