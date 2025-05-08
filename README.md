# llm-pmbok-finetuning
Este projeto transforma o conteúdo do PMBOK (PDF) em dados no formato Instruct para fine-tuning de modelos de linguagem.

## Preparar o ambiente
Você vai precisar de:
Python 3.10+
GPU com pelo menos 16 GB de VRAM (ou usar Colab/Paperspace)

## Instalar dependências:
```bash
pip install transformers datasets peft accelerate bitsandbytes
```

## Requisitos
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install openai tqdm
```

## Uso

1. Coloque seu arquivo `PMBOK.pdf` na raiz do projeto.
2. Execute o script:

```bash
cd scripts
python pdf_to_instruct.py
```

## Rodar o treinamento

Na raiz do projeto:

```bash
python train.py
```

## Executar
No terminal:

```bash
python inference.py
```
