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

MAFDET legge il tuo messaggio mentre lo scrivi, e **comincia a ragionarci prima che tu lo invii**.<br>
Premi Invio, e gran parte del lavoro è già fatta.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-EEEAE2?labelColor=0E0D0C)](LICENSE)
[![Runtime: local](https://img.shields.io/badge/runtime-local-D4892A?labelColor=0E0D0C)](#requisiti)
[![Status: prototype](https://img.shields.io/badge/status-prototype-6A655D?labelColor=0E0D0C)](#roadmap)

[English](README.md)

</div>

---

## Perché

Le interfacce di chat fanno aspettare il modello finché non premi Invio, e solo allora si mette al lavoro. Mentre scrivi, la GPU resta ferma. Quando invii, il modello deve prima *leggere* tutto il messaggio (prefill). Poi un modello che ragiona, come Qwen3 in modalità ragionamento, *pensa* per centinaia di token prima della prima parola della risposta. Su una GPU di fascia media quel ragionamento è l'attesa più lunga di tutte: dieci secondi e più su un problema, minuti con un modello più grande della VRAM.

MAFDET usa per entrambe le cose il tempo in cui scrivi. **Legge** il messaggio mentre lo scrivi, così all'Invio il prefill è già fatto. E **ragiona in anticipo**: il modello comincia a pensare al tuo messaggio mentre lo stai ancora scrivendo, e conserva quel ragionamento quando cambi il testo. Su Qwen3-4B, con 24 problemi digitati in tempo reale, l'attesa media dopo l'Invio passa da **12,2 s a 3,2 s**, con le stesse 24 risposte corrette su 24; su Qwen3-14B da **108 s a 47 s**.

MAFDET è pensato per l'hardware che la maggior parte delle persone possiede davvero. La macchina di riferimento è una **RTX 2060 con 6 GB di VRAM**, e tutto funziona anche **solo su CPU**.

> Il nome viene da Mafdet, la dea felina dell'antico Egitto detta "la corritrice", che colpiva serpenti e scorpioni prima che potessero attaccare.

## Come funziona

### Lettura in anticipo

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

### Ragionamento in anticipo

La lettura in anticipo toglie il prefill dall'attesa; il ragionamento però resta. Così il modello **ragiona anche mentre scrivi**: a blocchi brevi, sul messaggio così com'è in quel momento. Quando cambi il testo, il ragionamento non viene buttato. Resta nel prompt come appunti dopo la nuova versione del messaggio, il che costa solo un prefill, e il modello continua da lì. All'Invio finisce di ragionare da dove era arrivato e risponde. Su Qwen3-4B circa due terzi del ragionamento sono fatti prima dell'Invio. Come sono impostati gli appunti, e cosa ha cambiato, è in [Ragionamento in anticipo, nel dettaglio](#ragionamento-in-anticipo-nel-dettaglio) più sotto.

La lettura in anticipo è esatta. Il ragionamento in anticipo no, perché il modello ragiona su versioni del messaggio non ancora definitive: per questo è stato misurato su risposte verificabili.

## Avvio rapido

**Su Windows, con un doppio clic.** Scarica MAFDET (su GitHub, *Code → Download ZIP*, ed estrai lo zip; oppure `git clone https://github.com/TheLastOxygen/mafdet.git`) e fai doppio clic su **`start_mafdet.bat`**. La prima volta crea un ambiente Python in `.venv` e ci installa MAFDET: ci vuole circa un minuto, e serve [Python](https://www.python.org/downloads/) 3.10 o successivo. Dopo, un doppio clic apre MAFDET in meno di un secondo, e dopo un aggiornamento installa da solo le librerie nuove. Il terminale mostra man mano cosa sta scaricando: ogni libreria, e poi llama.cpp e i modelli che scegli nell'app, con dimensione, barra di avanzamento, velocità e tempo rimanente.

**Da terminale:**

```bash
git clone https://github.com/TheLastOxygen/mafdet.git
cd mafdet
python -m venv .venv          # Windows: py -3.12 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e .
mafdet
```

In entrambi i casi niente PyTorch e niente download a mano: MAFDET chiede se aprire la **finestra** o l'**app nel terminale**, e al primo avvio una procedura guidata fa il resto. Guarda il tuo computer, installa da sola la build giusta di llama.cpp (CUDA per una scheda NVIDIA, CPU altrimenti) e scarica il modello che rende di più sul tuo hardware.

<p align="center">
  <img src="assets/screenshots/gui-chat.png" alt="La finestra di MAFDET: il modello ha ragionato mentre il messaggio veniva scritto (455 token di ragionamento su 571) e ha risposto 1,6 s dopo l'invio" width="49%">
  <img src="assets/screenshots/gui-models.png" alt="Il catalogo dei modelli: per ogni modello, dove gira su questo computer, la velocità e l'attesa dopo l'invio con e senza MAFDET" width="49%">
</p>

## L'app

La stessa app esiste in due forme, sullo stesso motore:

| | Comando | Cos'è |
|---|---|---|
| <img src="assets/icons/watch.svg" width="22" alt=""> | `mafdet gui` | **La finestra**: chat, modelli, hardware e impostazioni, in italiano, inglese, francese o tedesco |
| <img src="assets/icons/strike.svg" width="22" alt=""> | `mafdet tui` | **L'app nel terminale**: le stesse viste nel terminale, pensate per la tastiera (da F1 a F4) |
| <img src="assets/icons/race.svg" width="22" alt=""> | `mafdet models` | il catalogo dei modelli per questo computer (`--family`, `--download`) |
| <img src="assets/icons/awaken.svg" width="22" alt=""> | `mafdet doctor` | controlla GPU, memoria, llama.cpp e modelli; `--install` installa llama.cpp (`--cuda 12`, `13` o `cpu` per scegliere la build) |

`mafdet` da solo apre il launcher, che può ricordare la scelta (`mafdet --ask` lo mostra di nuovo).

**Chat.** Mentre scrivi, una riga di stato sotto il messaggio mostra cosa sta facendo il modello: quanto ha già letto e quanti token di ragionamento ha scritto in anticipo (il campo si illumina di ambra mentre ragiona). All'invio resta da calcolare solo la coda. Il ragionamento è mostrato chiuso sopra la risposta, e ogni risposta finisce con i suoi numeri: prima parola dopo l'invio, token letti mentre scrivevi, ragionamento fatto prima dell'invio, velocità.

**Modelli.** Cinque famiglie: **Qwen3**, **Llama 3**, **Gemma 3**, **Mistral** e **Phi-4**, da 0,6 a 32 miliardi di parametri, ognuna in più quantizzazioni. Con ogni modello MAFDET legge il messaggio mentre scrivi e, a una pausa, scrive la risposta in anticipo; Qwen3 in più **ragiona in anticipo**, l'unica famiglia di cui è stata misurata la cornice di ragionamento. Ogni modello del catalogo dice cosa fa sul *tuo* computer, prima di scaricarlo: se sta nella VRAM o gira in parte sulla CPU, quanto veloce scrive e legge e, per Qwen3, dopo quanto arriva la risposta dopo l'invio con e senza MAFDET (per le altre famiglie, quanto ci vuole a scrivere una risposta tipica). La stima viene dal tuo hardware (banda di GPU e RAM, VRAM, ripartizione dei layer di llama.cpp tra GPU e CPU) e sul PC di riferimento sta entro il 12% delle velocità misurate (Qwen3-4B 89,6 stimati / 87,1 misurati tok/s, 8B 16,2 / 16,2, 14B 5,1 / 4,9). Una volta scaricato un modello, **Misura** lo prova in mezzo minuto e i numeri veri sostituiscono la stima. Il modello che rende di più è segnato come *Il migliore per te*, e ogni famiglia ha il suo migliore. Per aggiungere un modello basta una riga in `tools/build_catalog.py`: legge da Hugging Face le dimensioni dei file e la forma del modello.

**Hardware e impostazioni.** La vista Hardware mostra GPU, CPU, memoria e il runtime llama.cpp, con un pulsante per installarlo o reinstallarlo. Le impostazioni della chat restano da un avvio all'altro (in `~/.mafdet`, oppure `$MAFDET_HOME`).

<p align="center">
  <img src="assets/screenshots/tui-models.png" alt="L'app nel terminale: il catalogo dei modelli come tabella" width="49%">
  <img src="assets/screenshots/launcher.png" alt="Il launcher: finestra o terminale" width="49%">
</p>

## Requisiti

- **Windows**, con una **GPU NVIDIA** e driver aggiornati (obiettivo di test: RTX 2060, 6 GB), oppure solo una CPU. Per **Linux** ci sono già le build di llama.cpp e il codice, ma non è ancora stato provato
- **Python 3.10 o successivo** (provato con la 3.12)
- Spazio su disco: circa 0,7 GB per llama.cpp con CUDA (60 MB senza), più i modelli, da 0,6 GB (Qwen3-0.6B) a 2,3 GB per il 4B consigliato sulle schede da 6-8 GB
- **16 GB di RAM** consigliati per l'esecuzione solo su CPU e per i modelli più grandi della VRAM

L'app non ha bisogno di PyTorch: i modelli li esegue llama.cpp, e transformers fornisce solo i tokenizer. PyTorch serve solo per le misure sul backend PyTorch (sotto): installalo prima dall'index CUDA, poi `pip install -e ".[research]"`.

## Ragionamento in anticipo, nel dettaglio

Nell'app è attivo di default per Qwen3 (Impostazioni, *Ragiona mentre scrivo*); da riga di comando è `mafdet strike`, con le opzioni qui sotto. Il modello **comincia a pensare al tuo messaggio mentre lo stai ancora scrivendo**. Usa Qwen3 in modalità ragionamento, quindi prima di rispondere il modello ragiona in un blocco nascosto (mostrato chiuso sopra la risposta). Quel ragionamento di solito è l'attesa più lunga, e `strike` ne sposta il più possibile prima dell'invio:

1. **Mentre scrivi**, il modello ragiona a blocchi brevi di 32 token, sul messaggio così com'è in quel momento, a parole intere (dopo uno spazio o un segno di punteggiatura, oppure a una pausa).
2. **Quando il testo cambia**, il ragionamento non viene buttato. Resta nel prompt come appunti, dopo la nuova versione del messaggio. Spostarlo lì costa solo un prefill: la fase veloce e parallela, circa 60 volte più rapida che riscrivere gli stessi token sul 14B con la RTX 2060. Una nuova riga degli appunti nomina le parole appena scritte, così il modello continua a raccogliere materiale su di esse invece di ricominciare.
3. **All'invio**, un'ultima frase dice al modello che il messaggio ora è completo. Finisce di ragionare da dove era arrivato e risponde.

Gli appunti sono un elenco di **materiale per la risposta**, aperto da una cornice: *"L'utente sta ancora scrivendo. Non mi chiederò cosa chiederà, né osserverò che il testo è incompleto. Mentre aspetto raccolgo materiale per la risposta: per ogni termine chiave, nome o numero già scritto, una riga con cos'è e i fatti che so, oppure un risultato parziale."* seguita da una riga di esempio presa da un altro messaggio. Ogni volta che scrivi altro, una nuova riga nomina le parole nuove: `- New words "finestra di contesto": `. Così da *"Quando finisce la finestra di contesto degli LLM?"* gli appunti contengono già cos'è una finestra di contesto e cosa significa LLM, e da un problema i risultati parziali (`3 × 12 = 36`) prima ancora che la domanda sia stata scritta.

La cornice conta. Senza, il modello spendeva gli appunti a osservare che il messaggio era incompleto e, dopo ogni modifica, ricopiava il suo ultimo paragrafo. Su sei messaggi digitati in tempo reale, la quota di frasi degli appunti che parlano dell'utente o del messaggio (invece che del contenuto) è scesa dall'86% con una cornice generica (*"ricava quello che puoi"*) al 41-54% con questo elenco, e al 17-26% con la riga di esempio e un logit bias che, durante il ragionamento in anticipo e solo lì, rende meno probabili parole come *user*, *incomplete*, *typing* (gli ID dei token vengono dal tokenizer del modello caricato, quindi funziona con qualsiasi modello). In modalità ragionamento si usa anche la presence penalty suggerita dagli autori di Qwen3, che tiene il 4B lontano dai loop in cui ripete una frase fino al limite di token.

**Se fai una pausa e invii il messaggio senza cambiarlo**, `strike` va oltre e scrive anche la risposta. Questa parte è esatta: la risposta vale solo per il testo esatto per cui è stata scritta, e qualsiasi modifica la scarta. Con `--no-think` (senza ragionamento) `strike` fa solo questo, e con la decodifica greedy la risposta è identica token per token a quella avviata dopo l'invio (verificato).

Il ragionamento in anticipo **non è esatto**: il modello ragiona su versioni intermedie del messaggio, quindi la risposta può essere diversa da quella che darebbe partendo da zero. Per questo è stato misurato su risposte verificabili (sotto).

| Opzione | Cosa fa |
|---|---|
| `--no-think` | niente ragionamento: risponde subito, e in anticipo scrive solo la risposta esatta |
| `--think-ahead N` | ragionamento più lungo conservato tra una modifica e l'altra (predefinito 1024 token) |
| `--ahead N` | token di risposta scritti in anticipo quando il testo non cambia (predefinito 512) |
| `--branches N` | sperimentale: il modello prova anche a indovinare come finirai il messaggio e, su uno slot tutto suo, ragiona sulla previsione; se ha indovinato il messaggio esatto, quel ragionamento (e la risposta, se ci è arrivato) è pronto all'Invio. 1 è il valore misurato, vedi sotto |
| `--recycle` | riusa le risposte scritte per le versioni precedenti del messaggio come bozze n-gram per la decodifica speculativa (+4% di velocità misurato con una digitazione naturale, +20% sul codice) |
| `--draft repo:file` | un modello piccolo della stessa famiglia propone token per la decodifica speculativa (nessun guadagno misurato sulla RTX 2060, vedi sotto) |

`mafdet watch` e `mafdet strike` aprono la finestra con il modello e le opzioni date da riga di comando (`--gguf repo:file`), invece delle impostazioni dell'app. `demo_live.py` (`start_demo.bat`) è la vista passo passo per il debug: token, cache, holdback e tempi, dal vivo nel terminale (`--auto` scrive da sola un esempio).

## Misurare e verificare

| | Comando | | Cosa fa |
|---|---|---|---|
| <img src="assets/icons/hunt.svg" width="22" alt=""> | `mafdet hunt` | la caccia | simula persone che scrivono e misura l'attesa dopo l'invio |
| <img src="assets/icons/race.svg" width="22" alt=""> | `mafdet race` | la corsa | misura il costo di prefill e decode sulla tua macchina |
| <img src="assets/icons/judge.svg" width="22" alt=""> | `mafdet judge` | il giudizio | confronta i risultati con un calcolo da zero: deve finire con 0 errori |
| <img src="assets/icons/awaken.svg" width="22" alt=""> | `mafdet awaken` | il risveglio | controlla PyTorch e la GPU, per il backend PyTorch |

Girano su llama.cpp con `--backend llama` (senza PyTorch), oppure su PyTorch/transformers con il predefinito `--backend hf` (`pip install -e ".[research]"` dopo aver installato PyTorch dall'index CUDA: da PyPI, su Windows, pip prenderebbe la versione solo CPU). `mafdet --help` è la guida completa, `mafdet <comando> --help` le opzioni di uno. Aggiungi `--device cpu` a qualunque comando per misurare il caso senza GPU. I risultati vengono salvati come file CSV in `results/` (si cambia con `--out`).

### Scenari di digitazione

`mafdet hunt` riproduce tre scenari (si scelgono con `--scenario`):

- **clean**: un messaggio scritto dall'inizio alla fine
- **messy**: refusi corretti con backspace, una parola cancellata e riscritta, un'aggiunta a metà testo, poca punteggiatura
- **paste**: un documento lungo incollato, seguito da una domanda scritta a mano. È il caso in cui il guadagno è massimo.

La digitazione non viene riprodotta in tempo reale. Ogni prefill viene eseguito davvero sul modello, e la sua durata misurata viene collocata sulla linea del tempo della digitazione. Se il modello resta indietro rispetto a chi scrive, il ritardo si accumula e viene sommato all'attesa finale, come in un sistema reale.

## Risultati

Tutto su una RTX 2060 (6 GB), llama.cpp, modelli Q4_K_M, digitazione in tempo reale (da 5 a 8 caratteri al secondo, qualche pausa naturale, invio subito dopo l'ultimo carattere, senza rileggere).

**Ragionamento in anticipo, Qwen3-4B** (sta nella VRAM, 83 tok/s): 24 problemi con una risposta numerica nota, scritti in italiano (script e dati: `research/thinking_ahead_4b.py`).

| | Senza ragionamento in anticipo | `mafdet strike` |
|---|---|---|
| Risposte corrette | 24/24 | 24/24 |
| Attesa mediana per la prima parola della risposta | 10,7 s | **2,9 s** |
| Attesa media | 12,2 s | **3,2 s** |
| Più veloce | | in 24 problemi su 24 |
| Ragionamento in totale | 944 token | **662 token** |
| Ragionamento rimasto dopo l'invio | 944 token | **244 token** |

Stessa precisione, meno ragionamento in totale (il materiale raccolto mentre scrivi evita al modello di ripartire da zero), e quasi due terzi fatti prima dell'invio. Le due cornici precedenti, misurate allo stesso modo: una cornice generica dava 22/24 e una mediana di 4,4 s; nessuna cornice dava 22/24, 6,9 s e il 70% di ragionamento in più.

**Risposta esatta in anticipo, Qwen3-4B, senza ragionamento** (pausa prima dell'invio, testo invariato):

| Risposta | Scritti prima dell'invio | Risposta completa a schermo dopo l'invio, strike | Senza strike |
|---|---|---|---|
| 300 token | 200 | 1,2 s | 3,5 s |
| 168 token | 168 (tutta) | istantanea | 1,9 s |

**Modelli più grandi.** Quello che il modello riesce a fare prima dell'invio è limitato dal tempo di digitazione × la sua velocità di scrittura:

| Modello | Sta in 6 GB? | Scrive | Ragionamento scritto mentre digiti un messaggio da 30 s |
|---|---|---|---|
| Qwen3-4B | sì | 83 tok/s | quasi tutto |
| Qwen3-8B | no, in parte sulla CPU | da 13 a 16 tok/s | da 130 a 280 token |
| Qwen3-14B | no, in parte sulla CPU | 5 tok/s | circa 160 token |

Anche così, sul 14B (6 problemi digitati in tempo reale, 6/6 corretti in entrambi i casi) il ragionamento in anticipo dimezza l'attesa media dopo l'invio: **da 108 s a 47 s**, con 164 token di ragionamento scritti prima dell'invio in media. Le singole prove variano molto (lo stesso problema ha richiesto 106 s una volta e 154 s un'altra), quindi 6 problemi danno l'ordine di grandezza, non una cifra precisa. La lettura in anticipo guadagna di più quanto più il modello è grande: il 14B legge un documento incollato da 3.000 token in circa 10 s, tutti nascosti mentre scrivi.

**Un modello piccolo che raccoglie per uno grande (`--gatherer`, sperimentale).** Per i modelli più grandi della VRAM abbiamo provato a far scrivere materiale mentre scrivi a un modello piccolo e veloce (Qwen3-0.6B, anche allenato apposta con `training/`), che il 14B legge all'invio. Sulla RTX 2060 non batte il 14B che ragiona in anticipo da solo:

| 14B, attesa media dopo l'invio | |
|---|---|
| Senza ragionamento in anticipo | 108 s |
| **Ragionamento in anticipo da solo** | **47 s** |
| Raccoglitore, Qwen3-0.6B non allenato | 60 s |
| Raccoglitore, Qwen3-0.6B allenato | 79-86 s |
| Materiale ideale scritto a mano (tutti i risultati parziali giusti), presentato come "di un assistente veloce" | 62 s |
| Lo stesso materiale ideale, presentato come ragionamento del modello | 49 s |

Anche il materiale perfetto pareggia soltanto il ragionamento in anticipo: il 14B lo legge, poi rifà e ricontrolla i calcoli, tranne sui problemi più facili (17-24 s invece di 37-63 s). Conta più come il materiale viene presentato che la sua qualità (da 62 s a 49 s). Domande aperte per chi vuole portare avanti questa ricerca: combinare le due cose (il modello grande ragiona in anticipo *e* legge il materiale come suo), insegnare al modello grande a fidarsi dei risultati parziali, oppure un raccoglitore dove il modello grande non riesce affatto a ragionare in anticipo (solo CPU).

**Indovinare il finale (`--branches`, sperimentale).** Il ragionamento che resta dopo l'Invio riguarda soprattutto la fine del messaggio, che arriva per ultima: in un problema, la domanda. Qwen3-4B spesso la indovina prima che tu la scriva: in un run, con il testo a *"…e una penna da 2"*, una delle sue previsioni era già il resto esatto del messaggio, *"euro. Quanti euro gli restano?"*. Così con `--branches 1` il modello prova anche a indovinare come finirai il messaggio (solo quando mancano poche parole, e solo quando due previsioni di fila coincidono), e su uno slot di llama-server tutto suo ragiona sul messaggio previsto come se l'avessi già inviato, mentre il ragionamento principale aspetta. Se poi invii esattamente quel messaggio, il suo ragionamento e la sua risposta sono stati scritti proprio per quel prompt e vengono presi così come sono. Stessi 24 problemi, due run ciascuno (48 messaggi; script e dati in `research/`):

| | Ragionamento in anticipo | `--branches 1` |
|---|---|---|
| Risposte giuste | 48/48 | 48/48 |
| Attesa media per la prima parola della risposta | 4,17 s | **3,51 s** |
| Attesa mediana | 2,98 s | 2,85 s |
| Il decimo più lento | 7,9 s | **5,3 s** |
| Messaggi indovinati esatti | | 14 su 48 |

La colonna del ragionamento in anticipo è stata rimisurata per questo confronto, lo stesso giorno e sullo stesso codice: 4,17 s contro i 3,2 s della tabella sopra, due run degli stessi 24 problemi differiscono di tanto. Un guadagno modesto, soprattutto sulle risposte più lente, e su 24 problemi non ancora significativo (più veloce in 16 su 24, test dei segni p = 0,15). Cosa non ha funzionato: adottare una previsione solo vicina (*"ha rimasto"* per *"gli restano"*), con una frase che chiede al modello di ricontrollare il ragionamento sul messaggio vero, ha dato 2 risposte sbagliate su 4: la previsione aveva un altro numero (32 per 41), o una parola che cambiava il problema (*"età attuale"*). Due rami insieme sono più lenti di nessuno (mediana 4,50 s contro 2,94 s su metà dei problemi): sulla RTX 2060 due sequenze scrivono 1,7 volte i token di una, quattro 2,2 volte, quindi ogni ramo rallenta gli altri.

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
- [x] L'app: finestra e terminale, catalogo dei modelli con le stime per il tuo hardware, installazione automatica di llama.cpp, primo avvio guidato, quattro lingue
- [x] Cinque famiglie di modelli: Qwen3, Llama 3, Gemma 3, Mistral, Phi-4
- [x] Un modello piccolo raccoglie mentre scrivi, uno grande risponde: provato e misurato (nessun guadagno sulla RTX 2060, vedi Risultati)
- [ ] Ragionamento in anticipo per i modelli più grandi della VRAM che batta il modello grande che ragiona da solo
- [ ] Ragionamento in anticipo oltre Qwen3, anche per i modelli di ragionamento con formati propri (DeepSeek-R1, gpt-oss): ognuno richiede di misurare la sua cornice
- [ ] Linux provato; macOS (Metal) e GPU AMD

## Ricerca correlata

- [LiveMind: Low-latency Large Language Models with Simultaneous Inference](https://arxiv.org/abs/2406.14319)
- [StreamingThinker: Large Language Models Can Think While Reading](https://arxiv.org/abs/2510.17238)
- [SHANKS: Simultaneous Hearing and Thinking for Spoken Language Models](https://arxiv.org/abs/2510.06917)
- [Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](https://arxiv.org/abs/2512.10931)

## Licenza

[Apache-2.0](LICENSE)
