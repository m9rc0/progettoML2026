# Progetto Machine Learning 2026

Utilizzo di una **Graph Neural Network (GNN)** per risolvere il problema della colorazione di grafi, un problema NP-hard di ottimizzazione combinatoria.

## Problema

Data un grafo $G = (V, E)$, assegnare a ogni nodo un colore scelto tra $c$ disponibili in modo che nessun arco colleghi due nodi dello stesso colore. L'obiettivo è trovare il numero minimo di colori necessari (numero cromatico) e una colorazione valida, ovvero con zero conflitti.

## Architettura del modello

Il modello `ColoraGrafo` è una GNN che prende in input la struttura del grafo e produce per ogni nodo una distribuzione di probabilità sui $c$ colori:

1. **Embedding per nodo** — ogni nodo viene mappato in uno spazio latente di dimensione `hidden=64` tramite `nn.Embedding`. Questo permette alla rete di imparare una rappresentazione ottimale per ciascun nodo indipendentemente dalle feature in input.
2. **Due layer SAGEConv** — aggregano l'informazione dei vicini (GraphSAGE con aggregazione *mean*), ciascuno seguito da ReLU.
3. **Layer lineare finale** — proietta le rappresentazioni nello spazio dei colori ($\mathbb{R}^c$).
4. **Softmax** — produce una distribuzione di probabilità sui $c$ colori per ogni nodo.

## Loss e metriche

L'addestramento è **non supervisionato**: non esistono etichette di colore, la rete impara direttamente dalla struttura del grafo. La loss combina due termini:

$$\mathcal{L} = \underbrace{\sum_{(i,j) \in E} \sum_k p_i^k \, p_j^k}_{\text{energia di Potts}} + \lambda \underbrace{\sum_i \sum_k p_i^k \log p_i^k}_{\text{termine entropico}}$$

- **Energia di Potts differenziabile**: penalizza le configurazioni in cui nodi adiacenti hanno alta probabilità sullo stesso colore. Minimizzarla spinge la rete verso colorazioni senza conflitti.
- **Termine entropico**: incoraggia distribuzioni più nette (bassa entropia), favorendo assegnamenti certi a un singolo colore piuttosto che distribuzioni uniformi.
- **Iperparametro $\lambda$**: bilancia i due termini (valore usato: `ll=0.05`).

Come metrica di valutazione si usa il **numero di conflitti** — archi in cui entrambi i nodi hanno lo stesso colore dopo argmax — che deve raggiungere 0 per una colorazione valida.

## Procedura di addestramento

Dato che il problema è non convesso e l'ottimizzazione può convergere a minimi locali, si adotta una strategia **ensemble**: il training viene ripetuto 50 volte per ogni valore di $c$, reinizializzando pesi e embedding. Per ogni $c$ si riportano media e deviazione standard dei conflitti, e la frequenza di colorazioni perfette (conflitti = 0).

Ottimizzatore: Adam con `lr=0.001`–`0.003`, fino a 400–1000 epoche.

## Grafi testati

| Grafo | Nodi | Archi | Baseline greedy (`largest_first`) |
|-------|------|-------|-----------------------------------|
| g11   | 25   | 160   | c = 7                             |
| g17   | 128  | 2113  | c = 32                            |

I grafi sono in formato DIMACS `.col` e vengono convertiti in oggetti PyTorch Geometric `Data`.

**Risultati su g11** (50 run, 400 epoche): la rete trova colorazioni perfette con frequenza crescente all'aumentare di $c$. Con $c=7$ (pari alla baseline greedy) si ottengono colorazioni perfette nel 56% dei casi; con $c=8$ nel 82%.

**Risultati su g17** (50 run, 500 epoche): grafo significativamente più denso, il problema è più difficile. Con $c=35$ si ottiene una colorazione perfetta solo nell'2% dei casi; i conflitti medi scendono al diminuire di $c$ in modo più lento rispetto a g11.

## Stack

PyTorch, PyTorch Geometric, NetworkX, Matplotlib
