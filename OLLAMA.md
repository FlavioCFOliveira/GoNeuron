# Usar Modelos GoNeuron no Ollama (GGUF)

Este guia explica como converter um modelo treinado no GoNeuron para o formato **GGUF** e importá-lo para o **Ollama**.

## 1. Exportar o Modelo do GoNeuron

Nos exemplos generativos (como o `poetry_generator`), o modelo já exporta um ficheiro `weights.bin` após o treino. No seu código Go, pode fazer isto chamando:

```go
err := model.SaveBinaryWeights("weights.bin")
```

## 2. Converter para GGUF (Python)

Poderá usar o script de conversão abaixo. Necessita de instalar a biblioteca `gguf`:

```bash
pip install gguf numpy
```

### Script de Conversão (`convert_goneuron.py`)

Crie um ficheiro chamado `convert_goneuron.py`:

```python
import numpy as np
import struct
from gguf import GGUFWriter

def load_gonn_weights(filename):
    with open(filename, "rb") as f:
        magic = f.read(4)
        if magic != b"GONN":
            raise ValueError("Ficheiro inválido (Magic mismatch)")
        version = struct.unpack("<i", f.read(4))[0]
        num_tensors = struct.unpack("<i", f.read(4))[0]

        tensors = {}
        for _ in range(num_tensors):
            name_len = struct.unpack("<i", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            rank = struct.unpack("<i", f.read(4))[0]
            shape = struct.unpack(f"<{rank}i", f.read(rank * 4))
            data_len = struct.unpack("<i", f.read(4))[0]
            # Inverter shape para o padrão GGUF (Column-major vs Row-major do GoNeuron)
            data = np.fromfile(f, dtype=np.float32, count=data_len).reshape(shape)
            tensors[name] = data
        return tensors

def convert_to_gguf(input_bin, output_gguf):
    weights = load_gonn_weights(input_bin)
    writer = GGUFWriter(output_gguf, "goneuron")

    # Adicionar Metadados
    writer.add_architecture("goneuron")
    writer.add_name("GoNeuron Generative Model")

    for name, data in weights.items():
        print(f"A adicionar tensor: {name} com shape {data.shape}")
        writer.add_tensor(name, data)

    writer.write_config_file()
    writer.close()
    print(f"Sucesso! Modelo guardado em: {output_gguf}")

if __name__ == "__main__":
    convert_to_gguf("weights.bin", "goneuron_model.gguf")
```

Corra o script:
```bash
python3 convert_goneuron.py
```

## 3. Criar o Modelfile do Ollama

Crie um ficheiro chamado `Modelfile`:

```dockerfile
FROM ./goneuron_model.gguf

# Definir o Template de Prompt (opcional)
TEMPLATE """
{{ .Prompt }}
"""

# Definir parâmetros
PARAMETER temperature 0.7
PARAMETER top_k 40
```

## 4. Importar para o Ollama

Corra o seguinte comando no terminal:

```bash
ollama create goneuron-poetry -f Modelfile
```

## 5. Correr o Modelo

```bash
ollama run goneuron-poetry
```

---

## Nota Técnica
O GoNeuron utiliza uma arquitetura customizada. Para que o Ollama consiga **executar** a inferência (não apenas ler os pesos), o motor `llama.cpp` (que suporta o Ollama) precisaria de ter a arquitetura do GoNeuron mapeada.

Atualmente, este processo serve para **interoperabilidade e arquivo**, sendo que a inferência otimizada com Metal deve ser feita preferencialmente através do próprio binário Go gerado.
