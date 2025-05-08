from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Caminho para o modelo treinado
MODEL_PATH = "./pmbok-model/checkpoint-7"

# Carregar modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Certificar-se de que está usando CPU ou MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

print("🤖 Modelo carregado. Digite sua pergunta (ou 'sair' para encerrar):")

while True:
    question = input("\n📝 Você: ")
    if question.lower() in ["sair", "exit", "quit"]:
        print("Encerrando.")
        break

    prompt = f"### Instrução:\n{question}\n\n### Resposta:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta_final = response.split("### Resposta:\n")[-1].strip()
    print(f"\n🤖 PMBOK LLM: {resposta_final}")
