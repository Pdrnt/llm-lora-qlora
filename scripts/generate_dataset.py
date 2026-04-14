import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, AuthenticationError, BadRequestError

load_dotenv()

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN = "atendimento para barbearias"
TOTAL_EXAMPLES = 60
TRAIN_SPLIT = 0.9
MODEL_NAME = "gpt-4o-mini"


def save_jsonl(file_path: Path, records: list[dict]) -> None:
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_records(records: list[dict]) -> list[dict]:
    valid_records = []

    for item in records:
        if not isinstance(item, dict):
            continue

        prompt = item.get("prompt")
        response = item.get("response")

        if not isinstance(prompt, str) or not isinstance(response, str):
            continue

        prompt = prompt.strip()
        response = response.strip()

        if not prompt or not response:
            continue

        valid_records.append({
            "prompt": prompt,
            "response": response
        })

    return valid_records


def split_dataset(records: list[dict], train_split: float = 0.9) -> tuple[list[dict], list[dict]]:
    random.shuffle(records)
    split_index = int(len(records) * train_split)
    train_records = records[:split_index]
    test_records = records[split_index:]
    return train_records, test_records


def build_messages(total_examples: int, domain: str) -> list[dict]:
    system_prompt = f"""
Você é um gerador de dataset sintético para fine-tuning de modelos de linguagem.

Tarefa:
Gerar exemplos no domínio "{domain}".

Formato obrigatório:
Retorne APENAS uma lista JSON válida.
Cada item da lista deve ser um objeto com exatamente estas chaves:
- "prompt"
- "response"

Regras:
- Gere exatamente {total_examples} exemplos.
- Os prompts devem parecer perguntas, solicitações ou instruções reais de usuários.
- As respostas devem ser claras, úteis, naturais e coerentes com o domínio.
- Varie os temas.
- Não repita exemplos.
- Não escreva explicações fora do JSON.
- Não use markdown.
"""

    user_prompt = f"""
Gere {total_examples} pares de prompt e response para o domínio "{domain}".
Retorne somente JSON válido.
"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def extract_json_content(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    return text


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY não encontrada. "
            "Crie um arquivo .env na raiz do projeto com:\n"
            "OPENAI_API_KEY=sua_chave_aqui"
        )

    client = OpenAI(api_key=api_key)

    print("Gerando dataset sintético...")

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.8,
            messages=build_messages(TOTAL_EXAMPLES, DOMAIN),
        )
    except AuthenticationError as error:
        raise RuntimeError(
            "Falha de autenticação com a OpenAI. "
            "Verifique se a chave da API está correta."
        ) from error
    except RateLimitError as error:
        raise RuntimeError(
            "Limite ou cota da API excedido. "
            "Verifique o plano, billing ou saldo disponível da conta."
        ) from error
    except BadRequestError as error:
        raise RuntimeError(
            "A requisição para a API foi rejeitada. "
            "Verifique o nome do modelo e os parâmetros enviados."
        ) from error
    except APIError as error:
        raise RuntimeError(
            "Erro de API da OpenAI durante a geração do dataset."
        ) from error

    content = completion.choices[0].message.content or ""
    content = extract_json_content(content)

    try:
        records = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "A resposta do modelo não veio em JSON válido. "
            "Revise o prompt ou tente novamente."
        ) from error

    if not isinstance(records, list):
        raise ValueError("A resposta do modelo deve ser uma lista JSON.")

    records = validate_records(records)

    if len(records) < 50:
        raise ValueError(
            f"Foram gerados apenas {len(records)} exemplos válidos. "
            "O mínimo exigido pelo laboratório é 50."
        )

    train_records, test_records = split_dataset(records, TRAIN_SPLIT)

    train_path = OUTPUT_DIR / "train.jsonl"
    test_path = OUTPUT_DIR / "test.jsonl"

    save_jsonl(train_path, train_records)
    save_jsonl(test_path, test_records)

    print("Dataset gerado com sucesso.")
    print(f"Domínio: {DOMAIN}")
    print(f"Modelo usado: {MODEL_NAME}")
    print(f"Total de exemplos válidos: {len(records)}")
    print(f"Treino: {len(train_records)}")
    print(f"Teste: {len(test_records)}")
    print(f"Arquivo de treino: {train_path}")
    print(f"Arquivo de teste: {test_path}")


if __name__ == "__main__":
    main()