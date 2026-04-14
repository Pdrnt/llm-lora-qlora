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

env
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

O script scripts/infer.py foi adicionado para testar o adaptador treinado após o fine-tuning.

Execução
python scripts/infer.py
Entrega

A entrega do laboratório deve ser feita em repositório no GitHub contendo o código-fonte, os arquivos .jsonl e a documentação do projeto. A versão final deve ser marcada com a tag v1.0.

## Observação sobre uso de IA

Partes geradas/complementadas com IA, revisadas por Pedro Lima.

No código, houve auxílio de IA especificamente em:
- definição da estrutura inicial do projeto e organização dos arquivos;
- implementação da leitura da variável de ambiente `OPENAI_API_KEY` no `scripts/generate_dataset.py`;
- tratamento de erros de autenticação, cota e resposta inválida no `scripts/generate_dataset.py`;
- validação dos registros gerados e salvamento dos arquivos `train.jsonl` e `test.jsonl` no `scripts/generate_dataset.py`;
- configuração da quantização com `BitsAndBytesConfig` no `scripts/train_qlora.py`;
- configuração do `LoraConfig` com `task_type="CAUSAL_LM"`, `r=64`, `lora_alpha=16` e `lora_dropout=0.1` no `scripts/train_qlora.py`;
- definição inicial dos `TrainingArguments`, incluindo `paged_adamw_32bit`, `cosine` e `warmup_ratio=0.03` no `scripts/train_qlora.py`;
- estrutura inicial do pipeline de treinamento com `SFTTrainer` no `scripts/train_qlora.py`;
- apoio na estrutura do script `scripts/infer.py` para carregamento do modelo base, adaptador e geração de resposta.

As decisões finais, revisões e ajustes foram realizados por Pedro Lima.
