
# JND-LIC Portraits 
Nikola Bakic EE58/2022

Implementacija ključnih komponenti **JND-LIC** metode (Pan et al., 2025) zasnovana na **CompressAI** baseline-u (`Cheng2020Anchor`), sa evaluacijom na portretima iz **CelebA-HQ** skupa.

Cilj je integracija percepcione mere *Just Noticeable Difference* (JND) u proces učene kompresije slika, tako da se bitovi efikasnije raspoređuju na regione koje ljudski vizuelni sistem stvarno primećuje. Pošto autori originalnog rada nisu objavili izvorni kod, ovaj projekat predstavlja **samostalnu reimplementaciju ključnih komponenti** sa eksperimentalnom evaluacijom.

> **Napomena:** Projekat je rađen kao semestralni rad iz predmeta *TV i video tehnologije*, FTN UNS, 2025/26. Originalna tema (Tema 30) bila je prevod i objašnjenje rada; pristup je proširen na implementaciju i evaluaciju zbog nedostupnosti zvaničnog koda.

## Sadržaj

* [Motivacija](#motivacija)
* [Metoda](#metoda)
* [Struktura repozitorijuma](#struktura-repozitorijuma)
* [Instalacija](#instalacija)
* [Pokretanje](#pokretanje)
* [Skup podataka](#skup-podataka)
* [Eksperimenti](#eksperimenti)
* [Rezultati](#rezultati)
* [Diskusija](#diskusija)
* [Reference](#reference)
* [Licenca](#licenca)

## Motivacija

Klasični *learned image compression* (LIC) modeli optimizuju rate–distortion gubitak nad pikselskim metrikama (MSE, MS-SSIM), što ne odgovara uvek ljudskoj percepciji. JND modeli kvantifikuju prag iznad kojeg posmatrač primećuje izobličenje; ugrađivanjem JND mape u tok kompresije moguće je smanjiti bitsku brzinu bez vidljive degradacije kvaliteta — naročito na portretima, gde su koža, oči i usta perceptivno osetljivi.

## Metoda

Implementirane komponente:

1. **JND Estimator** (Eq. 1 iz rada) — klasični model po Wu et al. (2016):
   `J = LA + VM − C · min(LA, VM)`, `C = 0.3`
   gde je `LA` luminance adaptation (Chou & Li, 1995), `VM` visual masking (Sobel gradijent, koeficijent 0.117), računato nad Y kanalom (BT.601 konverzija).

2. **JND-FTM modul** (Eq. 2–4) — *Feature Transform Module* sa dve kapije:
   * Forget gate: `w = σ(conv([f_I, f_J]))`, `M_fg = 1 − w`
   * Memory gate: `O_mg = tanh(conv(cat(F_mask, f_J))) * w`
   * Izlaz: `f*_I = O_mg − f_I ⊙ M_fg`

3. **JND-QM mehanizam** (Eq. 5) — dinamička kvantizacija vođena JND mapom:
   * Trening: `y_I + noise * y_J_safe` (uniformni šum)
   * Inferencija: `ŷ_Iq = round(y_I / ŷ_J) * ŷ_J`

4. **Hyperprior pipeline za y_J (verzija v2)** — pun hyperprior model nad y_J granom (`h_a_J`, `h_s_J`, `EntropyBottleneck_J_z`, `GaussianConditional_J`), koji smanjuje y_J overhead sa **2.37 bpp na 0.32 bpp** (poboljšanje od 7.3×) u odnosu na inicijalnu v1 verziju.

5. **CompressAI baseline** — `Cheng2020Anchor` (quality=3) kao polazna tačka za fine-tuning. Težine `g_a`, `g_s`, `h_a`, `h_s`, kontekstnog modela i Gausovog uslovnog modela su zamrznute; treniraju se samo JND moduli i novi hyperprior pipeline za y_J.

6. **Trening petlja sa dva loss-a** — Adam optimizator sa StepLR scheduler-om, RD loss u dve varijante:
   * **MSE-only**: `λ · 255² · MSE + bpp`
   * **MSE+LPIPS**: `λ · 255² · (MSE + 1000 · LPIPS) + bpp` (po preporuci Pan et al.)

## Struktura repozitorijuma

```
.
├── jnd_lic_portraits.ipynb   # glavni notebook (kod, treniranje, evaluacija)
├── requirements.txt          # Python zavisnosti
├── .gitignore
├── LICENSE                   # MIT
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

1. Provera GPU okruženja i instalacija zavisnosti
2. Definicija JND Estimator-a, JND-FTM i JND-QM modula
3. Definicija `JND_LIC_v2` modela (sa hyperprior-om za y_J)
4. Učitavanje pretreniranog `Cheng2020Anchor` baseline-a
5. Preuzimanje i priprema CelebA-HQ portreta (256×256, split 180/20)
6. Trening petlje (MSE-only i MSE+LPIPS varijante, 50 epoha)
7. Evaluacija (PSNR, LPIPS, bpp) na celom validacionom skupu
8. Poređenje sa PWL JND-LC modelima (Pakdaman et al., 2024)
9. Ablaciona analiza komponenti
10. Vizuelna analiza rekonstrukcija i JND mapa

## Skup podataka

* **CelebA-HQ 256×256** — preuzima se iz HuggingFace huba (`korexyz/celeba-hq-256x256`).
* Korišćen podset: 200 slika, split 180 trening / 20 validacija.
* Augmentacije: `RandomCrop(256)`, `RandomHorizontalFlip`.
* Trening / validacioni split se generiše unutar notebook-a; nema potrebe za ručnim preuzimanjem.

## Eksperimenti

Trenirano je **6 modela** ukupno, organizovani u dve serije sa po tri λ vrednosti za RD krivu:

| Serija | λ vrednosti | Loss funkcija | Trajanje treninga |
|---|---|---|---|
| MSE-only | 0.004, 0.016, 0.032 | `MSE + bpp` | ~6 min/model na T4 |
| MSE+LPIPS | 0.004, 0.016, 0.032 | `MSE + 1000·LPIPS + bpp` | ~6 min/model na T4 |

Konfiguracija treninga:
* 50 epoha, batch size 4
* Adam optimizator (lr=1e-4 za glavne parametre, lr=1e-3 za aux quantiles)
* StepLR scheduler (gamma=0.5, step=15 epoha)
* Gradient clipping na normu 1.0
* Trainable: 4.80M / 16.64M parametara (28.9%)

Pored naših modela, u poređenje su uključeni:
* **Baseline cheng2020-anchor** iz CompressAI biblioteke (quality 1–6)
* **PWL JND-LC** (Pakdaman et al., 2024) — drugi JND-bazirani pristup, korišćeni javno dostupni pretrenirani modeli autora

## Rezultati

### RD performanse na validacionom skupu (20 slika)

| Metoda | bpp | PSNR (dB) | LPIPS |
|---|---|---|---|
| Baseline cheng2020 q=3 | 0.221 | 30.47 | 0.156 |
| Baseline cheng2020 q=6 | 0.795 | 35.01 | 0.041 |
| PWL JND-LC Q=3 | 0.215 | 30.44 | 0.163 |
| PWL JND-LC Q=6 | 0.772 | 35.12 | 0.044 |
| **JND-LIC v2 MSE λ=0.016** | 1.465 | 26.80 | 0.307 |
| **JND-LIC v2 MSE+LPIPS λ=0.016** | 1.859 | 25.88 | **0.128** |

### Glavni nalazi

1. **Arhitektura je verno reimplementirana.** Ablaciona konfiguracija „bez FTM i QM" reprodukuje rezultate baseline-a (PSNR 30.47 dB, LPIPS 0.155). Razlika u bpp-u (0.59 vs 0.22) potiče isključivo od y_J overhead-a.

2. **JND mapa korektno detektuje perceptualno bitne regione.** Visoke vrednosti se pojavljuju na ivicama i u teksturisanim oblastima (kosa, ivice odeće), dok su flat regioni (čelo, obrazi, pozadina) prepoznati kao tolerantni na izobličenje.

3. **Hyperprior pipeline za y_J je ključan.** Smanjenje overhead-a sa 2.37 bpp (v1) na 0.32 bpp (v2) — poboljšanje od 7.3 puta.

4. **Trade-off između PSNR i LPIPS.** MSE+LPIPS varijanta drastično poboljšava LPIPS metriku (sa 0.30 na 0.13), ali subjektivni vizuelni kvalitet je u nekim slučajevima lošiji — slike postaju tamnije i gube kontrast. LPIPS može biti zavarana modelima koji pogađaju distribuciju karakteristika u feature space-u AlexNet-a.

5. **PWL JND-LC pokazuje marginalna poboljšanja.** Čak i kompletno trenirani JND-bazirani model (Pakdaman et al.) ne nadmašuje baseline značajno na PSNR/LPIPS metrikama. Pan et al. koriste FID/KID kao primarne metrike — korist od JND pristupa nije primarno vidljiva preko PSNR/LPIPS.

## Diskusija

### Zašto rezultati ne nadmašuju baseline

Naša JND-LIC implementacija ne dostiže rezultate originalnog rada zbog:

1. **Drastično skraćen trening:** 50 epoha vs 200, 1× T4 GPU vs 8× RTX 3090, 180 slika vs 20.745.
2. **Zamrznut baseline:** g_a, g_s, h_a, h_s su trenirani isključivo MSE distortion target-om bez perceptualnog signala. JND moduli operišu nad već fiksiranim feature space-om.
3. **JND moduli kao bottleneck:** dodatni slojevi (FTM, QM) bez dovoljnog treninga gube informaciju umesto da je čuvaju.

### Pozitivni aspekti implementacije

* Arhitektura je u potpunosti ispravna (potvrđeno ablacijom).
* JND mapa funkcioniše prema specifikaciji rada.
* Hyperprior pipeline za y_J u v2 verziji demonstrira jasnu metodološku korisnost.
* Poređenje sa PWL JND-LC postavlja rezultate u širi kontekst literature.

### Mogući pravci poboljšanja

* Finije podešavanje težinskog odnosa u MSE+LPIPS loss-u (umesto fiksne vrednosti 1000)
* Zajednički trening JND modula i odmrznutih delova baseline-a
* Duži trening sa većim datasetom
* Implementacija potpuno paralelnih grana za sliku i JND mapu
* Evaluacija preko FID i KID metrika (primarne metrike u Pan et al.)

## Reference

* **Pan et al., 2025** — *JND-LIC: Learned Image Compression via Just Noticeable Difference for Human Visual Perception.* IEEE Transactions on Broadcasting, vol. 71, no. 1, pp. 217–228. [Repozitorijum autora (prazan)](https://github.com/TJU-Videocoding/JND-LIC)
* **Pakdaman et al., 2024** — *Perceptual Learned Image Compression via End-to-End JND-Based Optimization.* IEEE ICIP. [Repozitorijum sa pretreniranim modelima](https://github.com/sanaznami/JND-LC)
* **Wu et al., 2016** — *Enhanced Just Noticeable Difference Model for Images with Pattern Complexity.* IEEE TIP.
* **Cheng et al., 2020** — *Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules.* CVPR.
* **CompressAI** — Bégaint et al., 2020. [github.com/InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI)
* **LPIPS** — Zhang et al., 2018. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR.
* **CelebA-HQ** — Karras et al., 2018.

## Licenca

Distribuirano pod MIT licencom. Pogledati [LICENSE](LICENSE) za detalje.
