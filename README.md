# Laboratório 07 - Especialização de LLMs com LoRA e QLoRA

Projeto da disciplina com geração de dataset sintético, fine-tuning com LoRA/QLoRA e entrega versionada no GitHub.

## Estrutura do projeto
- `data/`: arquivos `.jsonl` de treino e teste
- `scripts/`: scripts de geração de dados, treinamento e inferência
- `outputs/`: saídas do treinamento e adaptador salvo

## Geração do dataset
O dataset sintético é gerado via API a partir do script `scripts/generate_dataset.py`, produzindo pares de `prompt` e `response` no domínio de atendimento para barbearias. Os dados são divididos em treino e teste e salvos em formato `.jsonl`.

### Configuração
Criar um arquivo `.env` na raiz do projeto com:

```env
OPENAI_API_KEY=sua_chave_aqui
Execução
python scripts/generate_dataset.py
Treinamento com QLoRA

O treinamento é realizado no script scripts/train_qlora.py.

Configurações implementadas no projeto:

quantização em 4 bits com load_in_4bit=True
tipo de quantização nf4
bnb_4bit_compute_dtype=torch.float16
LoraConfig com task_type="CAUSAL_LM"
r=64
lora_alpha=16
lora_dropout=0.1
optim="paged_adamw_32bit"
lr_scheduler_type="cosine"
warmup_ratio=0.03
Execução
python scripts/train_qlora.py
Inferência