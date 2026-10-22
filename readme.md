# Filter Generated Data

## age-detection

Versione moderna del tutorial PyImageSearch (OpenCV age detection):

| Passo | 2020 (tutorial) | Qui |
|---|---|---|
| Volti | Caffe SSD res10 | **YuNet** (OpenCV `FaceDetectorYN`, ONNX) |
| Età | Caffe CNN, 8 fasce | **ViT** Hugging Face (9 fasce), sostituibile con `--model` |

### Uso
```bash
pixi install
pixi run detect foto.jpg                 # rileva volti + età
pixi run detect cartella_foto/
pixi run crops cartella_crop/            # crop già pronti: salta il detector
pixi run detect foto.jpg --model <repo_hf>   # altro modello di età
```
Output: `output/results.csv` + immagini annotate in `output/`.
Primo avvio: scarica YuNet in `models/` e il modello HF (serve rete).

Note: l'output è una **fascia d'età** con probabilità; per un'età numerica
usare un modello di regressione (es. MiVOLO) passando `--model` se compatibile.
