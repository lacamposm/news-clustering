from pathlib import Path


def create_structure_medallion():
    """
    Crea la estructura de directorios medallion (bronze, silver, gold) dentro del directorio 'data'.
    """
    
    base_path = Path(__file__).resolve().parent.parent / "data"
    base_path.mkdir(parents=True, exist_ok=True)
    subcarpetas = ["bronze", "silver", "gold"]

    for subcarpeta in subcarpetas:
        (base_path / subcarpeta).mkdir(parents=True, exist_ok=True)


create_structure_medallion()
