import os
import yaml

MLRUNS_DIR = "mlruns"
LINKS_DIR = "mlruns_links"

os.makedirs(LINKS_DIR, exist_ok=True)

for exp_id in os.listdir(MLRUNS_DIR):
    exp_path = os.path.join(MLRUNS_DIR, exp_id)
    meta_path = os.path.join(exp_path, "meta.yaml")
    
    if not os.path.isdir(exp_path) or not os.path.isfile(meta_path):
        continue

    with open(meta_path) as f:
        meta = yaml.safe_load(f)
        exp_name = meta.get("name", None)
    
    if not exp_name:
        continue

    # Limpiamos el nombre por si hay caracteres raros
    clean_name = exp_name.replace(" ", "_").replace("/", "_")
    symlink_path = os.path.join(LINKS_DIR, clean_name)

    # Creamos el symlink si no existe
    try:
        if not os.path.exists(symlink_path):
            os.symlink(os.path.abspath(exp_path), symlink_path)
            print(f"✔️  {clean_name} -> {exp_path}")
    except Exception as e:
        print(f"⚠️  No se pudo crear link para {exp_name}: {e}")
