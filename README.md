# 📊 Visualizador de Sinais — Y-Balance & Step-Down

App em Streamlit para sincronizar e analisar sinais de Kinem (câmera) com
acelerômetro/giroscópio de celular posicionados em três segmentos — **L5**,
**Coxa** e **Tornozelo** — durante testes de Y-Balance e Step-Down.

## Funcionalidades

- Upload de múltiplos arquivos CSV/TXT (Kinem + ACC/GYR de cada segmento)
- Sincronização automática por detecção de pico + correlação cruzada
- Pré-processamento: detrend e filtro passa-baixa (Butterworth)
- Visualização de todos os eixos X, Y, Z sincronizados
- Checagem de qualidade (Kinem vs. celular, sobreposto em z-score)
- **Ângulo do joelho**: estimado pelo celular (filtro complementar ACC+GYR
  entre Coxa e Tornozelo) comparado ao ângulo ótico real do Kinem
  (geometria 3D Trocânter→Côndilo→Tornozelo)
- Exportação da janela selecionada para Excel (.xlsx)

## Rodando localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Suba este repositório no GitHub (repositório: `dropdown-analysis`).
2. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com sua conta GitHub.
3. Clique em **"New app"**, selecione o repositório `callegaribi-maker/dropdown-analysis`, branch `main` e arquivo principal `app.py`.
4. Clique em **Deploy** — em alguns minutos você terá um link público tipo
   `https://seu-app.streamlit.app`.

## Estrutura

- `app.py` — interface Streamlit (upload, sincronização, gráficos, exportação)
- `signal_utils.py` — funções puras de processamento de sinal (sem dependência do Streamlit)
- `requirements.txt` — dependências Python
