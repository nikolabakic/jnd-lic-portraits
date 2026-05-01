# JND-LIC Portraits

Implementacija ključnih komponenti **JND-LIC** metode (Pan et al., 2025) zasnovana na **CompressAI** baseline-u (`Cheng2020Anchor`), sa evaluacijom na portretima iz **CelebA-HQ** skupa.

Cilj je integracija percepcione mere *Just Noticeable Difference* (JND) u proces učene kompresije slika, tako da se bitovi efikasnije raspoređuju na regione koje ljudski vizuelni sistem stvarno primećuje.

## Sadržaj

- [Motivacija](#motivacija)
- [Metoda](#metoda)
- [Struktura repozitorijuma](#struktura-repozitorijuma)
- [Instalacija](#instalacija)
- [Pokretanje](#pokretanje)
- [Skup podataka](#skup-podataka)
- [Rezultati](#rezultati)
- [Ograničenja i naredni koraci](#ograničenja-i-naredni-koraci)
- [Reference](#reference)
- [Licenca](#licenca)

## Motivacija

Klasični *learned image compression* (LIC) modeli optimizuju rate–distortion gubitak nad pikselskim metrikama (MSE, MS-SSIM), što ne odgovara uvek ljudskoj percepciji. JND modeli kvantifikuju prag iznad kojeg posmatrač primećuje izobličenje; ugrađivanjem JND mape u tok kompresije moguće je smanjiti bitsku brzinu bez vidljive degradacije kvaliteta — naročito na portretima, gde su koža i oči perceptivno osetljivi.

## Metoda

Implementirane komponente:

1. **JND Estimator** — klasični model po Wu et al. (2016):
   `J = LA + VM − C · min(LA, VM)`
   gde je `LA` luminance adaptation, `VM` visual masking, a `C` parametar preklapanja.
2. **JND-FTM kvantizator** (Feature-to-Measurement) — kvantizacioni modul koji koristi JND prag kao gradient bottleneck u latentnom prostoru.
3. **CompressAI baseline** — `Cheng2020Anchor` (quality=3), pretrenirani checkpoint kao polazna tačka za fine-tuning.
4. **Hyperprior integracija** — proširenje sa dodatnim entropijskim modelom nad `y_J` granom radi smanjenja bitske brzine.
5. **Trening petlja** — Adam optimizator, ReduceLROnPlateau scheduler, `λ`-balansirani rate–distortion loss.

## Struktura repozitorijuma

```
.
├── jnd_lic_portraits.ipynb   # glavni notebook (kod, treniranje, evaluacija)
├── requirements.txt          # Python zavisnosti
├── .gitignore
├── LICENSE
└── README.md
```

## Instalacija

Preporučuje se Python ≥ 3.10 i CUDA-kompatibilan GPU (testirano na NVIDIA T4 / Google Colab).

```bash
git clone https://github.com/nikolabakic/jnd-lic-portraits.git
cd jnd-lic-portraits
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Za reprodukciju u Colab okruženju, učitaj notebook direktno na [colab.research.google.com](https://colab.research.google.com/) — sve `pip install` ćelije i preuzimanje skupa podataka su deo notebook-a.

## Pokretanje

```bash
jupyter lab jnd_lic_portraits.ipynb
```

Notebook je strukturiran sekvencijalno:

1. Provera GPU okruženja
2. Učitavanje pretreniranog `Cheng2020Anchor`
3. Definicija JND estimator-a i FTM kvantizatora
4. Preuzimanje i priprema CelebA-HQ portreta (256×256)
5. Trening petlja (40 epoha, batch=8)
6. Evaluacija (PSNR, MS-SSIM, bpp)
7. Vizuelna analiza rekonstrukcija

## Skup podataka

- **CelebA-HQ 256×256** — preuzima se iz HuggingFace huba (`korexyz/celeba-hq-256x256`).
- Trening / validacioni split se generiše unutar notebook-a; nema potrebe za ručnim preuzimanjem.
- Augmentacije: `RandomCrop(256)`, `RandomHorizontalFlip`, `RandomRotation`.

## Rezultati

| Model                              | PSNR (dB) | bpp   |
|------------------------------------|-----------|-------|
| Cheng2020Anchor (baseline)         | 39.05     | 0.055 |
| JND-LIC (40 epoha fine-tuning)     | ≈ 41.90   | ≈ 0.040 (y) + 2.30 (y_J) |

Tokom 40 epoha na T4 GPU dobijen je porast od **+2.85 dB PSNR** uz smanjenje bitske brzine glavnog kanala. Trening: ~1.1 min po 10 epoha. Konvergencija: ~0.07 dB/epoha, ~0.04 bpp/epoha.

## Ograničenja i naredni koraci

- `y_J` grana trenutno koristi jednu `EntropyBottleneck` instancu, što daje ~2.3 bpp dodatnog overhead-a; originalni rad koristi paralelni hyperprior model. **Sledeći korak:** ugraditi pun hyperprior za `y_J` (λ=0.016).
- Pojednostavljeno deljenje težina između paralelnih grana (autorski model koristi nezavisne težine).
- Trenutno se evaluira PSNR i MS-SSIM; planirano je dodavanje **LPIPS** percepcione metrike radi konzistentnosti sa originalnim radom.
- Proširenje na više `quality` nivoa (1–6) i poređenje sa BPG / VTM kodecima.

## Reference

- **Pan et al., 2025** — *JND-LIC: Learned Image Compression via Just Noticeable Difference for Human Visual Perception.* [Repozitorijum autora](https://github.com/TJU-Videocoding/JND-LIC)
- **Wu et al., 2016** — *Enhanced Just Noticeable Difference Model for Images with Pattern Complexity.* IEEE TIP.
- **Cheng et al., 2020** — *Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules.* CVPR.
- **CompressAI** — Bégaint et al., 2020. [github.com/InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI)
- **CelebA-HQ** — Karras et al., 2018.

## Licenca

Distribuirano pod MIT licencom. Pogledati [LICENSE](LICENSE) za detalje.
