import os
import sys
import importlib.util
from pathlib import Path

def verify_all_imports():
    root_dir = Path(__file__).resolve().parent
    
    # Ajouter la racine et le dossier 'src' au sys.path pour simuler l'exécution globale
    sys.path.insert(0, str(root_dir))
    if (root_dir / "").exists():
        sys.path.insert(0, str(root_dir / ""))

    python_files = list(root_dir.glob("**/*.py"))
    
    errors = 0
    checked = 0
    
    print(f"🔍 Analyse de {len(python_files)} fichiers Python...\n")

    for file_path in python_files:
        # Ignorer ce script lui-même et les environnements virtuels
        if file_path == Path(__file__).resolve() or ".venv" in file_path.parts or "venv" in file_path.parts:
            continue
            
        checked += 1
        relative_path = file_path.relative_to(root_dir)
        
        # Déterminer un nom de module unique pour l'importation
        module_name = ".".join(relative_path.with_suffix("").parts)
        
        try:
            # Tenter de charger le module et d'exécuter ses imports
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # sys.modules enregistre le module pour gérer les imports internes
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except (ModuleNotFoundError, ImportError) as e:
            print(f"❌ Erreur d'importation dans : {relative_path}")
            print(f"   👉 {type(e).__name__}: {e}\n")
            errors += 1
        except Exception:
            # On ignore les erreurs d'exécution pure (ex: variables manquantes au runtime),
            # on ne cible ici que les erreurs structurelles d'import.
            pass

    print("--- Rapport de vérification ---")
    if errors == 0:
        print(f"✅ Succès ! {checked} fichiers vérifiés. Aucun problème d'import détecté.")
    else:
        print(f"⚠️ Alerte : {errors} fichier(s) contiennent des imports cassés sur {checked} fichiers vérifiés.")
        sys.exit(1)

if __name__ == "__main__":
    verify_all_imports()