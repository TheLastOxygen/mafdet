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
| <img src="assets/icons/strike.svg" width="22" alt=""> | `mafdet strike` | il colpo | inizia a ragionare mentre scrivi, e scrive la risposta quando ti fermi | disponibile |
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
git clone https://github.com/TheLastOxygen/mafdet.git
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

Funziona con llama.cpp: estrai una [release di llama.cpp](https://github.com/ggml-org/llama.cpp/releases) in `./llama.cpp` (oppure imposta `MAFDET_LLAMA_DIR`). Il primo avvio scarica Qwen3-4B in Q4_K_M (circa 2,5 GB). Su Windows basta un doppio clic su `avvia_mafdet.bat` (apre `mafdet strike`, qui sotto). Con `--browser` la chat si apre nel browser predefinito invece che nella sua finestra.

### Ragionamento in anticipo

```bash
mafdet strike
```

`mafdet strike` è la stessa chat, un passo più in là: il modello **comincia a pensare al tuo messaggio mentre lo stai ancora scrivendo**. Usa Qwen3 in modalità ragionamento, quindi prima di rispondere il modello ragiona in un blocco nascosto (mostrato chiuso sopra la risposta). Quel ragionamento di solito è l'attesa più lunga, e `strike` ne sposta il più possibile prima dell'invio:

1. **Mentre scrivi**, il modello ragiona a blocchi brevi di 32 token, sul messaggio così com'è in quel momento, a parole intere (dopo uno spazio o un segno di punteggiatura, oppure a una pausa).
2. **Quando il testo cambia**, il ragionamento non viene buttato. Resta nel prompt come appunti, dopo la nuova versione del messaggio. Spostarlo lì costa solo un prefill: la fase veloce e parallela, circa 60 volte più rapida che riscrivere gli stessi token sul 14B con la RTX 2060. Una breve frase dice al modello che il messaggio è cambiato, così lo rilegge e prosegue da quello che aveva già ricavato.
3. **All'invio**, un'ultima frase dice al modello che il messaggio ora è completo. Finisce di ragionare da dove era arrivato e risponde.

Gli appunti cominciano con una cornice: *"L'utente sta ancora scrivendo... Non indovinerò il resto: ricaverò quello che posso dalla parte già scritta."* Senza, il modello spendeva gli appunti a osservare che il messaggio era incompleto e, dopo ogni modifica, ricopiava il suo ultimo paragrafo (misurato: il 70% di ragionamento in più, e due risposte su 24 mai arrivate). Con la cornice fa lavoro parziale utile, per esempio `3 × 12 = 36` prima ancora che la domanda sia stata scritta.

**Se fai una pausa e invii il messaggio senza cambiarlo**, `strike` va oltre e scrive anche la risposta. Questa parte è esatta: la risposta vale solo per il testo esatto per cui è stata scritta, e qualsiasi modifica la scarta. Con `--no-think` (senza ragionamento) `strike` fa solo questo, e con la decodifica greedy la risposta è identica token per token a quella avviata dopo l'invio (verificato).

Il ragionamento in anticipo **non è esatto**: il modello ragiona su versioni intermedie del messaggio, quindi la risposta può essere diversa da quella che darebbe partendo da zero. Per questo è stato misurato su risposte verificabili (sotto).

| Opzione | Cosa fa |
|---|---|
| `--no-think` | niente ragionamento: risponde subito, e in anticipo scrive solo la risposta esatta |
| `--think-ahead N` | ragionamento più lungo conservato tra una modifica e l'altra (predefinito 1024 token) |
| `--ahead N` | token di risposta scritti in anticipo quando il testo non cambia (predefinito 512) |
| `--recycle` | riusa le risposte scritte per le versioni precedenti del messaggio come bozze n-gram per la decodifica speculativa (+4% di velocità misurato con una digitazione naturale, +20% sul codice) |
| `--draft repo:file` | un modello piccolo della stessa famiglia propone token per la decodifica speculativa (nessun guadagno misurato sulla RTX 2060, vedi sotto) |

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

Tutto su una RTX 2060 (6 GB), llama.cpp, modelli Q4_K_M, digitazione in tempo reale (da 5 a 8 caratteri al secondo, qualche pausa naturale, invio subito dopo l'ultimo carattere, senza rileggere).

**Ragionamento in anticipo, Qwen3-4B** (sta nella VRAM, 83 tok/s): 24 problemi con una risposta numerica nota, scritti in italiano.

| | Senza ragionamento in anticipo | `mafdet strike` |
|---|---|---|
| Risposte corrette | 22/24 | 22/24 |
| Attesa mediana per la prima parola della risposta | 10,2 s | **4,4 s** |
| Attesa media | 11,8 s | **6,2 s** |
| Più veloce | | in 20 problemi su 22 |
| Ragionamento in totale | 914 token | 907 token |
| Ragionamento rimasto dopo l'invio | 914 token | **462 token** |

Stessa qualità, stessa quantità di ragionamento, metà fatta prima dell'invio. In entrambe le colonne 2 risposte non sono mai arrivate (il 4B a volte ragiona in tondo fino al limite di token, con o senza ragionamento in anticipo).

**Risposta esatta in anticipo, Qwen3-4B, senza ragionamento** (pausa prima dell'invio, testo invariato):

| Risposta | Scritti prima dell'invio | Risposta completa a schermo dopo l'invio, strike | Senza strike |
|---|---|---|---|
| 300 token | 200 | 1,2 s | 3,5 s |
| 168 token | 168 (tutta) | istantanea | 1,9 s |

**Modelli più grandi.** Quello che il modello riesce a fare prima dell'invio è limitato dal tempo di digitazione × la sua velocità di scrittura:

| Modello | Sta in 6 GB? | Scrive | Ragionamento scritto mentre digiti un messaggio da 30 s |
|---|---|---|---|
| Qwen3-4B | sì | 83 tok/s | quasi tutto |
| Qwen3-8B | no, in parte sulla CPU | circa 13 tok/s | da 130 a 280 token |
| Qwen3-14B | no, in parte sulla CPU | 5 tok/s | 130 token |

Sull'8B e sul 14B il ragionamento è di 400-1500 token, quindi prima dell'invio se ne fa al massimo una parte, e nelle singole prove la differenza di lunghezza del ragionamento tra due campionamenti è più grande del guadagno (esempi: 8B da 32 s a 20 s su un problema, da 74 s a 83 s su un altro; 14B da 235 s a 146 s). La lettura in anticipo invece guadagna di più quanto più il modello è grande: il 14B legge un documento incollato da 3.000 token in circa 10 s, tutti nascosti mentre scrivi.

**Provate, e perché non sono attive di default:**

- `--recycle` (le risposte alle versioni precedenti del messaggio come bozze n-gram): +4% di velocità di scrittura con una digitazione naturale, +20% sul codice, primo token un po' più tardi.
- `--draft` (Qwen3-0.6B che propone token al 14B): sulla GPU toglie VRAM al 14B, che porta più layer sulla CPU (da 4,8 a 3,5 tok/s); sulla CPU guadagna l'8% con bozze da 4 token e dimezza la velocità di lettura.

## Roadmap

- [x] Motore di prefill incrementale con sincronizzazione sul prefisso comune e holdback dei token
- [x] Benchmark, simulazione della digitazione e verifiche di correttezza
- [x] CLI unificata `mafdet`
- [x] `mafdet watch`: la finestra della chat
- [x] Backend llama.cpp con modelli quantizzati (GGUF)
- [x] `mafdet strike`: ragionamento in anticipo mentre scrivi, conservato tra le modifiche; la risposta esatta in anticipo a una pausa
- [ ] Un modello piccolo ragiona mentre scrivi, uno grande risponde: ragionamento in anticipo per i modelli più grandi della VRAM
- [ ] Misure ripetute sui modelli più grandi della VRAM

## Ricerca correlata

- [LiveMind: Low-latency Large Language Models with Simultaneous Inference](https://arxiv.org/abs/2406.14319)
- [StreamingThinker: Large Language Models Can Think While Reading](https://arxiv.org/abs/2510.17238)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](https://arxiv.org/abs/2512.10931)

## Licenza

[Apache-2.0](LICENSE)
