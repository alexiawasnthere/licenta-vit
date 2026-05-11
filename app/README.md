# Aplicatie de testare pentru modelul `train_cnn`

Aplicatia ruleaza local, intr-un browser, si afiseaza predictiile modelului 3D CNN in timp real din webcam. Este gandita sa fie copiata pe alt calculator impreuna cu modelul antrenat.

## Structura recomandata

```text
app/
  app.py
  inference.py
  requirements.txt
  README.md
model/
  best_model_3d_cnn.keras
  class_names.json
```

Modelul si `class_names.json` pot sta oriunde pe calculator. Le indici prin argumente la pornire.

## Instalare

Din folderul `app`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pe Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pornire

Exemplu, daca modelul este intr-un folder `model` langa `app`:

```bash
python app.py --model ../model/best_model_3d_cnn.keras --classes ../model/class_names.json
```

Aplicatia va afisa o adresa locala de forma `http://127.0.0.1:7860`. Deschide adresa in browser si permite accesul la camera.

## Note

- Modelul antrenat cu `train_cnn.py` asteapta 37 de cadre RGB, redimensionate la 128x128 si normalizate in `[0, 1]`.
- Predictia apare dupa ce aplicatia strange primele 37 de cadre din webcam, apoi se actualizeaza continuu.
- Daca exista `run_config.json` langa model, aplicatia citeste automat `num_frames` si `img_size` din el. Altfel foloseste valorile implicite `37` si `128`.
