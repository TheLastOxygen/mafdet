<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mafdet-logo-dark.svg">
  <img src="assets/mafdet-logo.svg" alt="" width="112">
</picture>

<h1>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mafdet-wordmark-dark.svg">
  <img src="assets/mafdet-wordmark.svg" alt="MAFDET" height="30">
</picture>
</h1>

### You type. It already knows.

MAFDET legge il tuo messaggio mentre lo scrivi e prepara il tuo LLM locale a rispondere.<br>
Premi invio e la risposta parte subito.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-EEEAE2?labelColor=0E0D0C)](LICENSE)
[![Runtime: local](https://img.shields.io/badge/runtime-local-D4892A?labelColor=0E0D0C)](#requisiti)
[![Status: prototype](https://img.shields.io/badge/status-prototype-6A655D?labelColor=0E0D0C)](#roadmap)

[English](README.md)

</div>

---

## Perché

Le interfacce di chat fanno aspettare il modello finché non premi invio, e solo allora si mette al lavoro. Mentre scrivi, la GPU resta ferma. Quando invii, il modello deve prima *leggere* tutto il messaggio (prefill), e solo dopo può iniziare a rispondere.

Su hardware costoso la lettura è così veloce che nessuno se ne accorge. Su una GPU di fascia media, su un portatile o su una CPU, leggere un prompt lungo può richiedere secondi interi: ed è proprio questo che MAFDET nasconde.

MAFDET è pensato per l'hardware che la maggior parte delle persone possiede davvero. La macchina di riferimento è una **RTX 2060 con 6 GB di VRAM**, e tutto funziona anche **solo su CPU**.

> Il nome viene da Mafdet, la dea felina dell'antico Egitto detta "la corritrice", che colpiva serpenti e scorpioni prima che potessero attaccare.

## Come funziona

Un LLM elabora il testo in due fasi:

| Fase | Cosa succede | Velocità |
|---|---|---|
| **Prefill** | legge il prompt; tutti i token vengono elaborati in parallelo | veloce |
| **Decode** | genera la risposta, un token alla volta | lenta |

Mentre legge, il modello salva risultati intermedi per ogni token nella **KV cache**. Per un dato prefisso quei risultati non cambiano mai, quindi si possono calcolare in anticipo.

MAFDET tiene la KV cache allineata al testo **mentre scrivi**:

1. A ogni pausa, tokenizza il testo attuale.
2. Trattiene gli ultimi token, perché una parola che stai ancora scrivendo può cambiare il modo in cui il testo viene diviso in token.
3. Trova il **prefisso comune più lungo** tra i token nuovi e quelli già in cache, taglia la cache in quel punto e fa il prefill solo di ciò che è nuovo.
4. Quando premi invio, restano da elaborare solo pochi token.

**La correttezza non dipende da come scrivi.** Backspace, modifiche a metà testo, blocchi incollati, punteggiatura o spazi mancanti: il controllo sul prefisso comune riporta sempre la cache esattamente a ciò che produrrebbe un calcolo da zero. Le euristiche come l'holdback e il debounce influiscono solo su quanto lavoro si spreca, mai sul risultato.

## Comandi

Lancia `mafdet` senza argomenti per la tabella dei comandi, `mafdet --help` per la guida completa (cosa fa ogni comando, con esempi), oppure `mafdet <comando> --help` per le opzioni di uno di essi.

| | Comando | | Cosa fa | Stato |
|---|---|---|---|---|
| <img src="assets/icons/awaken.svg" width="22" alt=""> | `mafdet awaken` | il risveglio | controlla la GPU, scarica e carica il modello | disponibile |
| <img src="assets/icons/watch.svg" width="22" alt=""> | `mafdet watch` | la guardia | apre la finestra della chat: il modello legge mentre scrivi | disponibile |
| <img src="assets/icons/strike.svg" width="22" alt=""> | `mafdet strike` | il colpo | pensa in anticipo e risponde appena premi invio | *previsto* |
| <img src="assets/icons/hunt.svg" width="22" alt=""> | `mafdet hunt` | la caccia | simula persone che scrivono e misura l'attesa dopo l'invio | disponibile |
| <img src="assets/icons/race.svg" width="22" alt=""> | `mafdet race` | la corsa | misura il costo di prefill e decode sulla tua macchina | disponibile |
| <img src="assets/icons/judge.svg" width="22" alt=""> | `mafdet judge` | il giudizio | confronta i risultati con un calcolo da zero | disponibile |

Ogni comando accetta le opzioni comuni `--model`, `--device`, `--dtype` e `--threads`. Funziona anche `python -m mafdet`.

## Requisiti

- **GPU NVIDIA** con driver aggiornati (obiettivo di test: RTX 2060, 6 GB), oppure solo una CPU
- **Python 3.12** (vanno bene anche 3.11 e 3.14). PyTorch non ha ancora pacchetti per Python 3.15.
- Circa **7 GB di spazio su disco** (PyTorch + modello)
- **16 GB di RAM** consigliati per l'esecuzione solo su CPU

Modello predefinito: **Qwen3-1.7B in fp16**. Le GPU Turing come la 2060 non supportano nativamente bf16.

## Installazione

L'ordine conta: **prima PyTorch dall'index CUDA**, poi MAFDET.

> **PyTorch non è una dipendenza del pacchetto `mafdet`, ed è voluto.** Se lo installasse pip da PyPI, su Windows prenderebbe la versione solo CPU e la GPU resterebbe inutilizzata. Installalo tu dall'index CUDA, come qui sotto, prima di installare MAFDET.

```bash
git clone https://github.com/<your-username>/mafdet.git
cd mafdet

python -m venv .venv          # Windows: py -3.12 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Così il comando `mafdet` viene installato nell'ambiente virtuale (`pip install -e .` lo lascia collegato alla cartella dei sorgenti, quindi le modifiche hanno effetto subito).

## Avvio rapido

```bash
mafdet awaken --load    # controllo della GPU, download del modello (~3,5 GB), generazione di prova
mafdet watch            # la finestra della chat
mafdet judge            # deve finire con 0 errori
mafdet race             # costo di prefill e decode sul tuo hardware
mafdet hunt             # attesa con e senza prefill incrementale
```

### Chat

```bash
mafdet watch
```

`mafdet watch` apre la finestra della chat di MAFDET. Mentre scrivi, una riga di stato sotto il messaggio mostra quanto il modello ne ha già letto; quando ti fermi legge anche la fine del prompt, così all'invio resta un solo token da calcolare. La conversazione resta nella cache da un turno all'altro. Le risposte sono impaginate in Markdown e puoi scorrerle liberamente mentre vengono scritte.

Funziona con llama.cpp: estrai una [release di llama.cpp](https://github.com/ggml-org/llama.cpp/releases) in `./llama.cpp` (oppure imposta `MAFDET_LLAMA_DIR`). Il primo avvio scarica Qwen3-4B in Q4_K_M (circa 2,5 GB). Su Windows basta un doppio clic su `avvia_mafdet.bat`. Con `--browser` la chat si apre nel browser predefinito invece che nella sua finestra.

`demo_live.py` (`avvia_demo.bat`) è la vista passo passo per il debug: token, cache, holdback e tempi, dal vivo nel terminale.

### Misure

Aggiungi `--device cpu` a qualunque comando per misurare il caso senza GPU. I risultati vengono salvati come file CSV in `results/` (si cambia con `--out`).

### Scenari di digitazione

`mafdet hunt` riproduce tre scenari (si scelgono con `--scenario`):

- **clean**: un messaggio scritto dall'inizio alla fine
- **messy**: refusi corretti con backspace, una parola cancellata e riscritta, un'aggiunta a metà testo, poca punteggiatura
- **paste**: un documento lungo incollato, seguito da una domanda scritta a mano. È il caso in cui il guadagno è massimo.

La digitazione non viene riprodotta in tempo reale. Ogni prefill viene eseguito davvero sul modello, e la sua durata misurata viene collocata sulla linea del tempo della digitazione. Se il modello resta indietro rispetto a chi scrive, il ritardo si accumula e viene sommato all'attesa finale, come in un sistema reale.

## Risultati

*In arrivo: misure su una RTX 2060 e su CPU.*

Cosa aspettarsi: sui messaggi brevi il guadagno è piccolo in termini assoluti, perché il prefill di poche decine di token è già veloce. Il guadagno cresce con la lunghezza del messaggio ed è massimo quando si incollano documenti lunghi e su CPU. MAFDET nasconde il costo di *leggere* il prompt, non quello di *generare* la risposta.

## Roadmap

- [x] Motore di prefill incrementale con sincronizzazione sul prefisso comune e holdback dei token
- [x] Benchmark, simulazione della digitazione e verifiche di correttezza
- [x] CLI unificata `mafdet`
- [x] `mafdet watch`: la finestra della chat
- [ ] Backend llama.cpp con modelli quantizzati (GGUF)
- [ ] `mafdet strike`: ragionamento speculativo sull'input parziale, riletto come appunti quando premi invio

## Ricerca correlata

- [LiveMind: Low-latency Large Language Models with Simultaneous Inference](https://arxiv.org/abs/2406.14319)
- [StreamingThinker: Large Language Models Can Think While Reading](https://arxiv.org/abs/2510.17238)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](https://arxiv.org/abs/2512.10931)

## Licenza

[Apache-2.0](LICENSE)
