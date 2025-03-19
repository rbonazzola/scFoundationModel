import anndata
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import pandas as pd
import numpy as np
import os
import gzip
import scipy


h5ad_path = f"{os.getenv('HOME')}/data/scrna_seq_arabidopsis/scPlantDB/CRA004476.h5ad.gz"

with gzip.open(h5ad_path, "rb") as f:
    adata = anndata.read_h5ad(f)

expression_matrix = adata.X
if not scipy.sparse.issparse(expression_matrix):
    raise ValueError("El formato de la matriz no es disperso.")

# Convertir a formato CSC (acceso eficiente por columnas)
expression_matrix = expression_matrix.tocsc()

print(expression_matrix)

genes = adata.var_names.tolist()
expression_data = pd.DataFrame(expression_matrix, columns=genes)
print(expression_data.head())

bins = [0, 10, 30, 60, 100]  # Ajustar según el dataset
labels = ["low", "medium", "high", "very_high"]

for gene in genes:
    expression_data[gene] = pd.cut(expression_data[gene], bins=bins, labels=labels, include_lowest=True)

sequences = expression_data.apply(lambda row: [f"{gene}_{exp}" for gene, exp in row.items()], axis=1)

output_file = "rna_sequences.txt"
with open(output_file, "w") as f:
    for i in range(expression_matrix.shape[0]):  # Iterar por célula
        expression_values = expression_matrix[i, :]
        bin_indices = np.digitize(expression_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(labels) - 1)  # Evitar índices fuera de rango
        
        tokens = [f"{genes[j]}_{labels[bin_indices[j]]}" for j in range(len(genes))]
        
        f.write(" ".join(tokens) + "\n")

with open("rna_sequences.txt", "w") as f:
    for seq in sequences:
        f.write(" ".join(seq) + "\n")

tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Entrenar tokenizer en las secuencias RNA-seq
trainer = trainers.WordLevelTrainer(
    vocab_size=50000, 
    min_frequency=1, 
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.train(["rna_sequences.txt"], trainer)

tokenizer.save("scRNAseq_tokenizer.json")

test_sequence = " ".join(sequences.iloc[0])  # Tomamos una célula aleatoria
encoded = tokenizer.encode(test_sequence)
decoded = tokenizer.decode(encoded.ids)

print("Secuencia original:", test_sequence)
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
print("Decodificado:", decoded)

tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordLevelTrainer(
    vocab_size=50000, 
    min_frequency=1, 
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)
tokenizer.train([output_file], trainer)

# Guardar tokenizer entrenado
tokenizer.save("scRNAseq_tokenizer.json")