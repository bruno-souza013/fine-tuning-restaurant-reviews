from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

OPTIONS = {
    "1": ("Preparar dataset", ROOT / "src" / "prepare_dataset.py"),
    "2": ("Verificar formato", ROOT / "scripts" / "check_format.py"),
    "3": ("Verificar carregamento quantizado", ROOT / "scripts" / "check_model_loading.py"),
    "4": ("Avaliar modelo base", ROOT / "scripts" / "evaluate_baseline.py"),
    "6": ("Avaliar adapter QLoRA", ROOT / "scripts" / "evaluate_adapter.py"),
    "7": ("Inferência interativa", ROOT / "scripts" / "inference.py"),
}


def mostrar_menu() -> None:
    print("\n=== Pipeline QLoRA ===")
    for codigo in ("1", "2", "3", "4"):
        print(f"{codigo}. {OPTIONS[codigo][0]}")
    print("5. Treinar adapter QLoRA")
    for codigo in ("6", "7"):
        print(f"{codigo}. {OPTIONS[codigo][0]}")
    print("0. Sair")


def executar(script: Path, argumentos: list[str] | None = None) -> None:
    comando = [sys.executable, str(script)]
    if argumentos:
        comando.extend(argumentos)

    resultado = subprocess.run(comando, cwd=ROOT)
    if resultado.returncode != 0:
        print(f"\nA etapa falhou com código {resultado.returncode}.")
    else:
        print("\nEtapa concluída com sucesso.")


def executar_treinamento() -> None:
    resposta = input(
        "Executar smoke test primeiro? [S/n]: "
    ).strip().lower()
    argumentos = ["--smoke-test"] if resposta != "n" else []
    executar(ROOT / "src" / "treinar_qlora.py", argumentos)


def main() -> None:
    while True:
        mostrar_menu()
        escolha = input("\nEscolha uma opção: ").strip()

        if escolha == "0":
            print("Encerrando.")
            return

        if escolha == "5":
            executar_treinamento()
            continue

        opcao = OPTIONS.get(escolha)
        if opcao is None:
            print("Opção inválida.")
            continue

        _, script = opcao
        executar(script)


if __name__ == "__main__":
    main()
