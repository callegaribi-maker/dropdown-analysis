# Dropdown Analysis — Visualizador de Sinais Sincronizados

App Streamlit para visualizar sinais de cinemática (posição/velocidade/aceleração
por eixo X/Y/Z) e IMU (acelerômetro e giroscópio) de joelho/tornozelo, a partir
de um arquivo `.xlsx` com uma aba por região do corpo.

## O que o app faz

- Lê qualquer `.xlsx` com abas no formato: `Tempo (s)`, `<segmento> X/Y/Z`,
  `<segmento> v(X/Y/Z)`, `<segmento> a(X/Y/Z)`, `ACC_X/Y/Z`, `GYR_X/Y/Z`.
- Dropdowns para escolher: região do corpo (aba), dispositivo/tipo de sinal
  (Cinemática - Posição/Velocidade/Aceleração, IMU - Acelerômetro/Giroscópio) e eixo.
- Segmenta os ciclos de teste (repetições) a partir de uma coluna de referência
  configurável (padrão: coluna D da aba `L5`), com 3 métodos: picos, vales ou
  cruzamento por zero. O número de ciclos é detectado automaticamente (não é fixo).
- Exporta todos os gráficos (todas as combinações de região × dispositivo × eixo)
  como um `.zip` de PNGs.

## Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abra o link local (normalmente `http://localhost:8501`) e envie o arquivo `.xlsx`.

## Deploy no Streamlit Community Cloud

1. Suba este repositório para o GitHub.
2. Em https://share.streamlit.io, clique em "New app", selecione o repositório,
   branch `main` e o arquivo `app.py`.
3. Depois de publicado, envie o `.xlsx` diretamente pela interface web — os
   dados não ficam salvos no repositório, apenas processados em memória na sessão.

## Estrutura

```
dropdown-analysis/
├── app.py            # app Streamlit
├── requirements.txt  # dependências
└── README.md
```
